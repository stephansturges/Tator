import AppKit
import Foundation
import MLX
import MLXDINOv3

struct WorkerRequest: Decodable {
    let id: String?
    let type: String?
    let protocolVersion: Int?
    let inputPath: String?
    let imagePaths: [String]?
    let outputPath: String?
    let includePatchTokens: Bool?
    let includeLastHiddenState: Bool?

    enum CodingKeys: String, CodingKey {
        case id
        case type
        case protocolVersion = "protocol_version"
        case inputPath = "input_path"
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

func normalizedPixelBatch(path: String) throws -> MLXArray {
    let url = URL(fileURLWithPath: path)
    let arrays = try loadArrays(url: url)
    guard let pixels = arrays["pixels"] else {
        throw WorkerError.invalidRequest("missing_pixels_tensor")
    }
    guard pixels.ndim == 4, pixels.shape[0] > 0,
        pixels.shape[1] == 224, pixels.shape[2] == 224, pixels.shape[3] == 3
    else {
        throw WorkerError.invalidRequest("invalid_pixels_shape")
    }
    let mean = MLXArray(
        [Float(0.485), Float(0.456), Float(0.406)],
        [1, 1, 1, 3]
    )
    let std = MLXArray(
        [Float(0.229), Float(0.224), Float(0.225)],
        [1, 1, 1, 3]
    )
    return (pixels.asType(.float32) / Float(255.0) - mean) / std
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
                defer {
                    // Keep model weights resident, but return per-request Metal
                    // intermediates instead of allowing the shared cache to
                    // grow for the lifetime of a long analysis.
                    Memory.clearCache()
                }
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
                guard let outputPath = request.outputPath, !outputPath.isEmpty else {
                    throw WorkerError.invalidRequest("missing_output_path")
                }

                let startedAt = Date().timeIntervalSince1970
                let batch: MLXArray
                let transport: String
                if let inputPath = request.inputPath, !inputPath.isEmpty {
                    batch = try normalizedPixelBatch(path: inputPath)
                    transport = "safetensors_pixels_v2"
                } else if let imagePaths = request.imagePaths, !imagePaths.isEmpty {
                    var inputs: [MLXArray] = []
                    inputs.reserveCapacity(imagePaths.count)
                    for path in imagePaths {
                        inputs.append(try processor(loadImage(path: path)))
                    }
                    batch = concatenated(inputs, axis: 0)
                    transport = "image_paths_v1"
                } else {
                    throw WorkerError.invalidRequest("missing_input")
                }
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
                eval(arrays.values)

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
                    "transport": transport,
                    "batch_size": batch.shape[0],
                    "elapsed_seconds": Date().timeIntervalSince1970 - startedAt,
                    "memory": [
                        "active_bytes": Memory.activeMemory,
                        "cache_bytes": Memory.cacheMemory,
                        "peak_bytes": Memory.peakMemory,
                    ],
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
