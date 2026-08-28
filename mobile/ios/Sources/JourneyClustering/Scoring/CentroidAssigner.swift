import Foundation

/// Centroid Distance Calculator and Nearest Cluster Assigner for Swift/iOS.
public final class CentroidAssigner: @unchecked Sendable {
    public struct AssignmentResult: Sendable, Equatable {
        public let clusterId: Int64
        public let distance: Double

        public init(clusterId: Int64, distance: Double) {
            self.clusterId = clusterId
            self.distance = distance
        }
    }

    public let clusterIds: [Int64]
    public let centroids: [[Float]] // [n_clusters, embeddingDim]

    public init(clusterIds: [Int64], centroids: [[Float]]) {
        self.clusterIds = clusterIds
        self.centroids = centroids
    }

    public func assign(embedding: [Float]) -> AssignmentResult {
        var minDistance = Float.infinity
        var bestIndex = 0

        for cIdx in 0..<centroids.count {
            let centroid = centroids[cIdx]
            var sumSq: Float = 0.0
            let limit = min(embedding.count, centroid.count)
            for d in 0..<limit {
                let diff = embedding[d] - centroid[d]
                sumSq += (diff * diff)
            }
            let dist = sqrt(sumSq)
            if dist < minDistance {
                minDistance = dist
                bestIndex = cIdx
            }
        }

        let bestClusterId = !clusterIds.isEmpty ? clusterIds[bestIndex] : Int64(bestIndex)
        return AssignmentResult(clusterId: bestClusterId, distance: Double(minDistance))
    }

    public static func load(manifestJson: String, binaryCentroids: Data) throws -> CentroidAssigner {
        guard let manifestData = manifestJson.data(using: .utf8),
              let manifest = try JSONSerialization.jsonObject(with: manifestData) as? [String: Any],
              let clusterIdsRaw = manifest["cluster_ids"] as? [Any] else {
            throw NSError(domain: "CentroidAssigner", code: 1, userInfo: [NSLocalizedDescriptionKey: "Invalid manifest.json"])
        }

        let clusterIds = clusterIdsRaw.compactMap { item -> Int64? in
            if let num = item as? NSNumber { return num.int64Value }
            if let str = item as? String { return Int64(str) }
            return nil
        }
        let embeddingDim = manifest["embedding_dim"] as? Int ?? 48
        let nClusters = clusterIds.count

        var cursor = 0
        let bytesCount = binaryCentroids.count

        func readShort() -> UInt16 {
            guard cursor + 2 <= bytesCount else { return 0 }
            let val = binaryCentroids.withUnsafeBytes { ptr -> UInt16 in
                ptr.loadUnaligned(fromByteOffset: cursor, as: UInt16.self)
            }
            cursor += 2
            return UInt16(littleEndian: val)
        }

        var centroids = [[Float]](repeating: [Float](repeating: 0.0, count: embeddingDim), count: nClusters)
        for c in 0..<nClusters {
            for d in 0..<embeddingDim {
                centroids[c][d] = Float16Helper.halfToFloat(readShort())
            }
        }

        return CentroidAssigner(clusterIds: clusterIds, centroids: centroids)
    }
}
