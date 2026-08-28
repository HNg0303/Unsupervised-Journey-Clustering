import Foundation

/// Intermediate result from JourneyScorer before post-processing and business taxonomy enrichment.
public struct JourneyScoringOutput: Sendable {
    public let assignedClusterId: Int64
    public let nearestCentroidClusterId: Int64
    public let distanceToCentroid: Double
    public let geometricAnomaly: Bool
    public let markovLogprob: Double?
    public let generativeAnomaly: Bool
    public let severeAnomaly: Bool
    public let frictionFlags: [String]
    public let nextAction: NextActionPrediction?
    public let featureMatrix: JourneyFeatureMatrix

    public init(
        assignedClusterId: Int64,
        nearestCentroidClusterId: Int64,
        distanceToCentroid: Double,
        geometricAnomaly: Bool,
        markovLogprob: Double?,
        generativeAnomaly: Bool,
        severeAnomaly: Bool,
        frictionFlags: [String],
        nextAction: NextActionPrediction?,
        featureMatrix: JourneyFeatureMatrix
    ) {
        self.assignedClusterId = assignedClusterId
        self.nearestCentroidClusterId = nearestCentroidClusterId
        self.distanceToCentroid = distanceToCentroid
        self.geometricAnomaly = geometricAnomaly
        self.markovLogprob = markovLogprob
        self.generativeAnomaly = generativeAnomaly
        self.severeAnomaly = severeAnomaly
        self.frictionFlags = frictionFlags
        self.nextAction = nextAction
        self.featureMatrix = featureMatrix
    }
}

/// Core Scorer computing vector embeddings, Euclidean centroid distance, Markov companion logprob, and anomalies.
public final class JourneyScorer: @unchecked Sendable {
    private let vectorizer: JourneyVectorizer
    private let centroidAssigner: CentroidAssigner
    private let markov: MarkovBank
    private let thresholds: ScoreThresholds

    public init(
        vectorizer: JourneyVectorizer,
        centroidAssigner: CentroidAssigner,
        markov: MarkovBank,
        thresholds: ScoreThresholds = ScoreThresholds()
    ) {
        self.vectorizer = vectorizer
        self.centroidAssigner = centroidAssigner
        self.markov = markov
        self.thresholds = thresholds
    }

    /// Score a prepared journey sequence.
    public func score(prepared: PreparedJourney) -> JourneyScoringOutput {
        // 1. Vectorize into multi-channel feature matrix + projected embedding
        let matrix = vectorizer.transform(journey: prepared)

        // 2. Assign nearest cluster centroid
        let assignment = centroidAssigner.assign(embedding: matrix.projectedEmbedding)
        let distance = assignment.distance
        let geometricAnomaly = distance > thresholds.distanceP95
        let assignedCluster = geometricAnomaly ? -1 : assignment.clusterId

        // 3. Markov Companion Logprob
        let logprob = markov.score(
            tokens: prepared.tokens,
            clusterId: geometricAnomaly ? nil : assignment.clusterId
        )
        let generativeAnomaly = (logprob != nil && logprob! < thresholds.markovP05)
        let severeAnomaly = geometricAnomaly && (logprob != nil && logprob! < thresholds.markovP01)

        // 4. Behavioral Friction Flags
        var flags: [String] = []
        if prepared.backRate > thresholds.backRateP90 { flags.append("excessive_back") }
        if Double(prepared.nLoopRemoved) > thresholds.loopsP90 { flags.append("navigation_loop") }
        if prepared.revisitRatio > thresholds.revisitP90 { flags.append("screen_thrash") }
        if prepared.spanSeconds > thresholds.spanP95 { flags.append("slow_journey") }
        if geometricAnomaly { flags.append("unknown_archetype") }
        if generativeAnomaly { flags.append("improbable_transitions") }

        // 5. Next Action Prediction
        let next = markov.predictNext(
            tokens: prepared.tokens,
            clusterId: geometricAnomaly ? nil : assignment.clusterId
        )

        return JourneyScoringOutput(
            assignedClusterId: assignedCluster,
            nearestCentroidClusterId: assignment.clusterId,
            distanceToCentroid: distance,
            geometricAnomaly: geometricAnomaly,
            markovLogprob: logprob,
            generativeAnomaly: generativeAnomaly,
            severeAnomaly: severeAnomaly,
            frictionFlags: flags,
            nextAction: next,
            featureMatrix: matrix
        )
    }
}
