// swift-tools-version: 6.0
import PackageDescription

let package = Package(
    name: "TatorMLXDINOv3Worker",
    platforms: [
        .macOS(.v14),
    ],
    products: [
        .executable(name: "mlx-dinov3-worker", targets: ["MLXDINOv3Worker"]),
        .executable(name: "mlx-dinov3-convert", targets: ["MLXDINOv3Convert"]),
    ],
    dependencies: [
        .package(url: "https://github.com/vincentamato/MLXDINOv3.git", revision: "3122d7905cca21012b4c249e8ddad19ff78f54bc"),
        .package(url: "https://github.com/ml-explore/mlx-swift.git", from: "0.31.3"),
        .package(url: "https://github.com/huggingface/swift-transformers", from: "1.3.0"),
    ],
    targets: [
        .executableTarget(
            name: "MLXDINOv3Worker",
            dependencies: [
                "MLXDINOv3",
                .product(name: "MLX", package: "mlx-swift"),
            ],
            path: "Sources/MLXDINOv3Worker"
        ),
        .executableTarget(
            name: "MLXDINOv3Convert",
            dependencies: [
                .product(name: "MLX", package: "mlx-swift"),
                .product(name: "Hub", package: "swift-transformers"),
            ],
            path: "Sources/MLXDINOv3Convert"
        ),
    ]
)
