import Foundation

/// Business Archetype & Cluster Mapping Repository.
public final class ClusterMapping: @unchecked Sendable {
    private let archetypes: [Int64: ClusterArchetype]

    public init(json: String) throws {
        guard let data = json.data(using: .utf8),
              let obj = try JSONSerialization.jsonObject(with: data) as? [String: Any] else {
            throw NSError(domain: "ClusterMapping", code: 1, userInfo: [NSLocalizedDescriptionKey: "Invalid cluster_mapping.json"])
        }

        var map: [Int64: ClusterArchetype] = [:]
        for (k, v) in obj {
            guard let item = v as? [String: Any] else { continue }
            let cid: Int64
            if let cNum = item["cluster_id"] as? NSNumber {
                cid = cNum.int64Value
            } else if let cStr = item["cluster_id"] as? String, let parsed = Int64(cStr) {
                cid = parsed
            } else if let kParsed = Int64(k) {
                cid = kParsed
            } else {
                cid = -1
            }

            map[cid] = ClusterArchetype(
                clusterId: cid,
                clusterName: item["cluster_name"] as? String ?? "Archetype \(cid)",
                businessFamily: item["business_family"] as? String ?? "Chưa phân loại",
                businessSubmodule: item["business_submodule"] as? String ?? "",
                businessDetail: item["business_detail"] as? String ?? "",
                namingConfidence: item["naming_confidence"] as? String ?? "medium",
                namingSource: item["naming_source"] as? String ?? "",
                needsReview: item["needs_review"] as? Bool ?? false,
                journeyCount: item["journey_count"] as? Int ?? 0,
                journeyShare: (item["journey_share"] as? NSNumber)?.doubleValue ?? 0.0,
                medoidPath: item["medoid_path"] as? String ?? ""
            )
        }
        self.archetypes = map
    }

    public func lookup(clusterId: Int64?) -> ClusterArchetype? {
        guard let clusterId = clusterId else { return nil }
        return archetypes[clusterId] ?? archetypes[-1]
    }
}
