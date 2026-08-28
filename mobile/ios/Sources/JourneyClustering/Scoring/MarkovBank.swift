import Foundation

/// Markov Companion Model for Journey Logprob Scoring and Next Action Prediction.
public final class MarkovBank: @unchecked Sendable {
    public struct Edge: Sendable {
        public let token: String
        public let count: Double

        public init(token: String, count: Double) {
            self.token = token
            self.count = count
        }
    }

    public struct Bucket: Sendable {
        public let transitions: [String: [Edge]]
        public let totals: [String: Double]

        public init(transitions: [String: [Edge]], totals: [String: Double]) {
            self.transitions = transitions
            self.totals = totals
        }
    }

    private let smoothing: Double
    private let vocabSize: Int
    private let vocabulary: Set<String>
    private let clusters: [Int64: Bucket]
    private let global: Bucket

    public init(markovJson: String) throws {
        guard let data = markovJson.data(using: .utf8),
              let root = try JSONSerialization.jsonObject(with: data) as? [String: Any] else {
            throw NSError(domain: "MarkovBank", code: 1, userInfo: [NSLocalizedDescriptionKey: "Invalid markov.json"])
        }

        self.smoothing = root["smoothing"] as? Double ?? 0.5
        let vocabList = (root["vocabulary"] as? [String]) ?? []
        self.vocabSize = root["vocab_size"] as? Int ?? (!vocabList.isEmpty ? vocabList.count : 1000)
        self.vocabulary = Set(vocabList)

        let globalObj = (root["global"] as? [String: Any]) ?? [:]
        self.global = MarkovBank.parseBucket(globalObj)

        let clustersObj = (root["clusters"] as? [String: Any]) ?? [:]
        var cMap: [Int64: Bucket] = [:]
        for (k, v) in clustersObj {
            if let bObj = v as? [String: Any], let cId = Int64(k) {
                cMap[cId] = MarkovBank.parseBucket(bObj)
            }
        }
        self.clusters = cMap
    }

    public func score(tokens: [String], clusterId: Int64?) -> Double? {
        guard tokens.count >= 2 else { return nil }
        let bucket = (clusterId.flatMap { clusters[$0] }) ?? global
        let denominatorMass = smoothing * Double(max(vocabSize, 1))
        var totalLogprob = 0.0
        var pairs = 0

        for i in 0..<(tokens.count - 1) {
            let src = tokens[i]
            let dst = tokens[i + 1]
            if !vocabulary.isEmpty && (!vocabulary.contains(src) || !vocabulary.contains(dst)) {
                continue
            }

            let edges = bucket.transitions[src]
            let count = edges?.first(where: { $0.token == dst })?.count ?? 0.0
            let total = (bucket.totals[src] ?? 0.0) + denominatorMass

            totalLogprob += (log(count + smoothing) - log(total))
            pairs += 1
        }

        return pairs == 0 ? nil : (totalLogprob / Double(pairs))
    }

    public func predictNext(tokens: [String], clusterId: Int64?, topK: Int = 3) -> NextActionPrediction? {
        guard let src = tokens.last else { return nil }
        if !vocabulary.isEmpty && !vocabulary.contains(src) { return nil }
        let bucket = (clusterId.flatMap { clusters[$0] }) ?? global
        guard let edges = bucket.transitions[src], !edges.isEmpty else { return nil }

        var totalObserved = 0.0
        for edge in edges { totalObserved += edge.count }

        let denominator = (bucket.totals[src] ?? 0.0) + smoothing * Double(max(vocabSize, 1))
        guard let best = edges.first else { return nil }

        return NextActionPrediction(
            token: best.token,
            smoothedProbability: (best.count + smoothing) / denominator,
            observedShare: (totalObserved > 0) ? (best.count / totalObserved) : 0.0
        )
    }

    private static func parseBucket(_ obj: [String: Any]) -> Bucket {
        let totalsObj = (obj["totals"] as? [String: Any]) ?? [:]
        var totals: [String: Double] = [:]
        for (k, v) in totalsObj {
            if let num = v as? NSNumber { totals[k] = num.doubleValue }
        }

        let transObj = (obj["transitions"] as? [String: Any]) ?? [:]
        var transitions: [String: [Edge]] = [:]
        for (src, edgesVal) in transObj {
            var edgeList: [Edge] = []
            if let arr = edgesVal as? [[String: Any]] {
                for row in arr {
                    if let token = row["token"] as? String, let count = (row["count"] as? NSNumber)?.doubleValue {
                        edgeList.append(Edge(token: token, count: count))
                    }
                }
            } else if let dict = edgesVal as? [String: Any] {
                for (dst, countVal) in dict {
                    if let count = (countVal as? NSNumber)?.doubleValue {
                        edgeList.append(Edge(token: dst, count: count))
                    }
                }
            }
            transitions[src] = edgeList.sorted { $0.count > $1.count }
        }
        return Bucket(transitions: transitions, totals: totals)
    }
}
