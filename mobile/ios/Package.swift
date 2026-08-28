// swift-tools-version: 5.9
// The swift-tools-version declares the minimum version of Swift required to build this package.

import PackageDescription

let package = Package(
    name: "JourneyClustering",
    platforms: [
        .iOS(.v14),
        .macOS(.v11)
    ],
    products: [
        .library(
            name: "JourneyClustering",
            targets: ["JourneyClustering"]
        ),
    ],
    dependencies: [],
    targets: [
        .target(
            name: "JourneyClustering",
            dependencies: [],
            path: "Sources/JourneyClustering"
        ),
        .testTarget(
            name: "JourneyClusteringTests",
            dependencies: ["JourneyClustering"],
            path: "Tests/JourneyClusteringTests"
        ),
    ]
)
