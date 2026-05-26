import AppKit
import Foundation
import MLX
import MLXDINOv3

struct WorkerRequest: Decodable {
    let id: String?
    let type: String?
    let imagePaths: [String]?
    let outputPath: String?
    let includePatchTokens: Bool?
    let includeLastHiddenState: Bool?

    enum CodingKeys: String, CodingKey {
        case id
        case type
        case imagePaths = "image_paths"
        case outputPath = "output_path"
        case includePatchTokens = "include_patch_tokens"
        case includeLastHiddenState = "include_last_hidden_state"
    }
}

enum WorkerError: Error, CustomStringConvertible {
    case missingArgument(String)
    case imageLoadFailed(String)
    case invalidRequest(String)

    var description: String {
        switch self {
        case .missingArgument(let name):
            return "missing_argument:\(name)"
        case .imageLoadFailed(let path):
            return "image_load_failed:\(path)"
        case .invalidRequest(let reason):
            return "invalid_request:\(reason)"
        }
    }
}

func emit(_ payload: [String: Any]) {
    let data = try! JSONSerialization.data(withJSONObject: payload, options: [.sortedKeys])
    FileHandle.standardOutput.write(data)
    FileHandle.standardOutput.write(Data([0x0A]))
}

func stderr(_ message: String) {
    if let data = (message + "\n").data(using: .utf8) {
        FileHandle.standardError.write(data)
    }
}

func argumentValue(_ name: String) -> String? {
    let args = CommandLine.arguments
    for index in args.indices {
        if args[index] == name, index + 1 < args.count {
            return args[index + 1]
        }
    }
    return nil
}

func loadImage(path: String) throws -> NSImage {
    guard let image = NSImage(contentsOfFile: path) else {
        throw WorkerError.imageLoadFailed(path)
    }
    return image
}

@main
struct MLXDINOv3Worker {
    static func main() {
        do {
            try run()
        } catch {
            emit(["ok": false, "error": "\(error)"])
            exit(1)
        }
    }

    static func run() throws {
        guard let modelDir = argumentValue("--model-dir") else {
            throw WorkerError.missingArgument("--model-dir")
        }
        let model = try DinoVisionTransformer.loadPretrained(from: modelDir)
        let processor = ImageProcessor()
        emit(["ok": true, "ready": true])

        let decoder = JSONDecoder()
        decoder.keyDecodingStrategy = .useDefaultKeys

        while let line = readLine(strippingNewline: true) {
            if line.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
                continue
            }
            do {
                guard let data = line.data(using: .utf8) else {
                    throw WorkerError.invalidRequest("not_utf8")
                }
                let request = try decoder.decode(WorkerRequest.self, from: data)
                if request.type == "shutdown" {
                    emit(["ok": true, "shutdown": true])
                    return
                }
                guard let requestId = request.id, !requestId.isEmpty else {
                    throw WorkerError.invalidRequest("missing_id")
                }
                guard let imagePaths = request.imagePaths, !imagePaths.isEmpty else {
                    throw WorkerError.invalidRequest("missing_image_paths")
                }
                guard let outputPath = request.outputPath, !outputPath.isEmpty else {
                    throw WorkerError.invalidRequest("missing_output_path")
                }

                var inputs: [MLXArray] = []
                inputs.reserveCapacity(imagePaths.count)
                for path in imagePaths {
                    inputs.append(try processor(loadImage(path: path)))
                }
                let batch = concatenated(inputs, axis: 0)
                let outputs = model(batch)

                var arrays: [String: MLXArray] = [
                    "cls_token": outputs.clsToken.asType(.float32)
                ]
                if request.includePatchTokens ?? true {
                    arrays["patch_tokens"] = outputs.patchTokens.asType(.float32)
                }
                if request.includeLastHiddenState ?? false {
                    arrays["last_hidden_state"] = outputs.lastHiddenState.asType(.float32)
                }

                let outputURL = URL(fileURLWithPath: outputPath)
                try FileManager.default.createDirectory(
                    at: outputURL.deletingLastPathComponent(),
                    withIntermediateDirectories: true
                )
                try save(arrays: arrays, url: outputURL)

                var tensors: [String: Any] = [:]
                for (name, array) in arrays {
                    tensors[name] = ["shape": array.shape, "dtype": "float32"]
                }
                emit([
                    "id": requestId,
                    "ok": true,
                    "output_path": outputPath,
                    "tensors": tensors,
                ])
            } catch {
                stderr("MLX-DINOv3 worker request failed: \(error)")
                var requestId: String = ""
                if let data = line.data(using: .utf8),
                    let request = try? decoder.decode(WorkerRequest.self, from: data)
                {
                    requestId = request.id ?? ""
                }
                emit(["id": requestId, "ok": false, "error": "\(error)"])
            }
        }
    }
}
