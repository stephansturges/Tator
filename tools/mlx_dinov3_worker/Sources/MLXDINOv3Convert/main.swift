import Foundation
import Hub
import MLX

enum ConversionError: Error, CustomStringConvertible {
    case invalidConfig
    case unsupportedModelType(String)

    var description: String {
        switch self {
        case .invalidConfig:
            return "invalid_config"
        case .unsupportedModelType(let modelType):
            return "unsupported_model_type:\(modelType)"
        }
    }
}

func mapViTWeights(_ ptWeights: [String: MLXArray]) -> [String: MLXArray] {
    var mlxWeights: [String: MLXArray] = [:]
    for (ptKey, ptValue) in ptWeights {
        if ptKey == "embeddings.mask_token" {
            continue
        }
        if ptKey == "embeddings.patch_embeddings.weight" {
            mlxWeights[ptKey] = ptValue.transposed(0, 2, 3, 1)
        } else {
            mlxWeights[ptKey] = ptValue
        }
    }
    return mlxWeights
}

func convert(modelId: String, outputDir: String) async throws {
    let hub = HubApi()
    let repoURL = try await hub.snapshot(from: modelId, matching: ["config.json", "model.safetensors"])
    let configURL = repoURL.appendingPathComponent("config.json")
    let configData = try Data(contentsOf: configURL)
    guard let config = try JSONSerialization.jsonObject(with: configData) as? [String: Any] else {
        throw ConversionError.invalidConfig
    }
    let modelType = String(describing: config["model_type"] ?? "dinov3_vit")
    if modelType.lowercased().contains("convnext") {
        throw ConversionError.unsupportedModelType(modelType)
    }

    let weightsURL = repoURL.appendingPathComponent("model.safetensors")
    let ptWeights = try loadArrays(url: weightsURL)
    let mlxWeights = mapViTWeights(ptWeights)

    let outputURL = URL(fileURLWithPath: outputDir)
    try FileManager.default.createDirectory(at: outputURL, withIntermediateDirectories: true)
    let outputConfig = outputURL.appendingPathComponent("config.json")
    let outputWeights = outputURL.appendingPathComponent("model.safetensors")
    let savedConfig = try JSONSerialization.data(withJSONObject: config, options: [.prettyPrinted, .sortedKeys])
    try savedConfig.write(to: outputConfig)
    try save(arrays: mlxWeights, url: outputWeights)
}

@main
struct MLXDINOv3Convert {
    static func main() async {
        let args = CommandLine.arguments
        guard args.count >= 3 else {
            fputs("usage: mlx-dinov3-convert <huggingface-model-id> <output-dir>\n", stderr)
            exit(2)
        }
        do {
            try await convert(modelId: args[1], outputDir: args[2])
        } catch {
            fputs("conversion failed: \(error)\n", stderr)
            exit(1)
        }
    }
}
