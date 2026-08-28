import Foundation

// MARK: - 1. Event Models

/// Raw clickstream event parsed from production CSV or real-time SDK JSON stream.
public struct ClickstreamEvent: Codable, Sendable, Equatable {
    public let sourceIndex: Int
    public let eventId: String
    public let deviceId: String
    public let customerId: String?
    public let sessionId: String
    public let createdAt: String?
    public let timestamp: Int64
    public let key: String
    public let segment: String
    public let name: String
    public let screenId: String
    public let durationSeconds: Int
    public let visit: String?

    public init(
        sourceIndex: Int,
        eventId: String,
        deviceId: String,
        customerId: String? = nil,
        sessionId: String,
        createdAt: String? = nil,
        timestamp: Int64,
        key: String,
        segment: String,
        name: String,
        screenId: String = "",
        durationSeconds: Int = 0,
        visit: String? = nil
    ) {
        self.sourceIndex = sourceIndex
        self.eventId = eventId
        self.deviceId = deviceId
        self.customerId = customerId
        self.sessionId = sessionId
        self.createdAt = createdAt
        self.timestamp = timestamp
        self.key = key
        self.segment = segment
        self.name = name
        self.screenId = screenId
        self.durationSeconds = durationSeconds
        self.visit = visit
    }

    enum CodingKeys: String, CodingKey {
        case sourceIndex = "source_index"
        case eventId = "event_id"
        case deviceId = "device_id"
        case customerId = "customer_id"
        case sessionId = "session_id"
        case createdAt = "created_at"
        case timestamp
        case key
        case segment
        case name
        case screenId = "screen_id"
        case durationSeconds = "duration_seconds"
        case visit
    }
}

/// Normalized click with canonical paths, resolved screen context, and semantic tokens.
public struct CanonicalClick: Sendable, Equatable {
    public let raw: ClickstreamEvent
    public let eventType: String
    public let segmentName: String
    public let screenContext: String
    public let eventToken: String
    public let eventTimeMillis: Int64
    public let gapPrevSeconds: Double?
    public let gapNextSeconds: Double?
    public let semantics: SemanticLabel
    public let tokenL1: String
    public let tokenL2: String
    public let tokenL3: String
    public let operationToken: String

    public init(
        raw: ClickstreamEvent,
        eventType: String,
        segmentName: String,
        screenContext: String,
        eventToken: String,
        eventTimeMillis: Int64,
        gapPrevSeconds: Double?,
        gapNextSeconds: Double?,
        semantics: SemanticLabel = SemanticLabel(),
        tokenL1: String = "",
        tokenL2: String = "",
        tokenL3: String = "",
        operationToken: String = ""
    ) {
        self.raw = raw
        self.eventType = eventType
        self.segmentName = segmentName
        self.screenContext = screenContext
        self.eventToken = eventToken
        self.eventTimeMillis = eventTimeMillis
        self.gapPrevSeconds = gapPrevSeconds
        self.gapNextSeconds = gapNextSeconds
        self.semantics = semantics
        self.tokenL1 = tokenL1
        self.tokenL2 = tokenL2
        self.tokenL3 = tokenL3
        self.operationToken = operationToken
    }
}

/// Coarse business semantics resolved from the path taxonomy.
public struct SemanticLabel: Codable, Sendable, Equatable {
    public let businessFamily: String
    public let businessModule: String
    public let businessObject: String
    public let operation: String
    public let operationStage: String
    public let semanticConfidence: Double
    public let semanticEvidence: String

    public var isKnown: BooleanLiteralType {
        return businessFamily != "unknown"
    }

    public init(
        businessFamily: String = "unknown",
        businessModule: String = "unknown",
        businessObject: String = "unknown",
        operation: String = "unknown",
        operationStage: String = "unknown",
        semanticConfidence: Double = 0.0,
        semanticEvidence: String = ""
    ) {
        self.businessFamily = businessFamily
        self.businessModule = businessModule
        self.businessObject = businessObject
        self.operation = operation
        self.operationStage = operationStage
        self.semanticConfidence = semanticConfidence
        self.semanticEvidence = semanticEvidence
    }

    enum CodingKeys: String, CodingKey {
        case businessFamily = "business_family"
        case businessModule = "business_module"
        case businessObject = "business_object"
        case operation
        case operationStage = "operation_stage"
        case semanticConfidence = "semantic_confidence"
        case semanticEvidence = "semantic_evidence"
    }
}

// MARK: - 2. Journey Data Models

/// Segmented and cleaned journey sequence ready for feature extraction and scoring.
public struct PreparedJourney: Sendable {
    public let journeyId: String
    public let sessionId: String
    public let deviceId: String
    public let customerId: String?
    public let platform: String
    public let boundaryReason: String
    public let rawEvents: [ClickstreamEvent]
    public let cleanedEvents: [CanonicalClick]
    public let tokens: [String]
    public let channels: [String: [String]]
    public let numeric: [String: Double]
    public let nEventsFinal: Int
    public let nLoopRemoved: Int
    public let nDedupRemoved: Int
    public let backRate: Double
    public let revisitRatio: Double
    public let spanSeconds: Double

    public init(
        journeyId: String,
        sessionId: String,
        deviceId: String,
        customerId: String?,
        platform: String,
        boundaryReason: String,
        rawEvents: [ClickstreamEvent],
        cleanedEvents: [CanonicalClick],
        tokens: [String],
        channels: [String: [String]],
        numeric: [String: Double],
        nEventsFinal: Int,
        nLoopRemoved: Int,
        nDedupRemoved: Int,
        backRate: Double,
        revisitRatio: Double,
        spanSeconds: Double
    ) {
        self.journeyId = journeyId
        self.sessionId = sessionId
        self.deviceId = deviceId
        self.customerId = customerId
        self.platform = platform
        self.boundaryReason = boundaryReason
        self.rawEvents = rawEvents
        self.cleanedEvents = cleanedEvents
        self.tokens = tokens
        self.channels = channels
        self.numeric = numeric
        self.nEventsFinal = nEventsFinal
        self.nLoopRemoved = nLoopRemoved
        self.nDedupRemoved = nDedupRemoved
        self.backRate = backRate
        self.revisitRatio = revisitRatio
        self.spanSeconds = spanSeconds
    }
}

/// Input feature bundle for JourneyScorer / JourneyVectorizer.
public struct JourneyFeatures: Sendable {
    public let tokens: [String]
    public let numeric: [String: Double]
    public let channels: [String: [String]]

    public init(tokens: [String], numeric: [String: Double], channels: [String: [String]] = [:]) {
        self.tokens = tokens
        self.numeric = numeric
        self.channels = channels
    }
}

/// Multi-channel feature matrices and final projected embedding.
public struct JourneyFeatureMatrix: Sendable {
    public let primaryEmbedding: [Float]
    public let channelEmbeddings: [String: [Float]]
    public let scaledNumericBlock: [Float]
    public let concatenatedVector: [Float]
    public let projectedEmbedding: [Float]

    public init(
        primaryEmbedding: [Float],
        channelEmbeddings: [String: [Float]],
        scaledNumericBlock: [Float],
        concatenatedVector: [Float],
        projectedEmbedding: [Float]
    ) {
        self.primaryEmbedding = primaryEmbedding
        self.channelEmbeddings = channelEmbeddings
        self.scaledNumericBlock = scaledNumericBlock
        self.concatenatedVector = concatenatedVector
        self.projectedEmbedding = projectedEmbedding
    }
}

// MARK: - 3. Archetype & Threshold Models

/// Business mapping metadata for an archetype cluster from cluster_mapping.json.
public struct ClusterArchetype: Codable, Sendable, Equatable {
    public let clusterId: Int64
    public let clusterName: String
    public let businessFamily: String
    public let businessSubmodule: String
    public let businessDetail: String
    public let namingConfidence: String
    public let namingSource: String
    public let needsReview: Bool
    public let journeyCount: Int
    public let journeyShare: Double
    public let medoidPath: String

    public init(
        clusterId: Int64,
        clusterName: String,
        businessFamily: String = "Chưa phân loại",
        businessSubmodule: String = "",
        businessDetail: String = "",
        namingConfidence: String = "medium",
        namingSource: String = "",
        needsReview: Bool = false,
        journeyCount: Int = 0,
        journeyShare: Double = 0.0,
        medoidPath: String = ""
    ) {
        self.clusterId = clusterId
        self.clusterName = clusterName
        self.businessFamily = businessFamily
        self.businessSubmodule = businessSubmodule
        self.businessDetail = businessDetail
        self.namingConfidence = namingConfidence
        self.namingSource = namingSource
        self.needsReview = needsReview
        self.journeyCount = journeyCount
        self.journeyShare = journeyShare
        self.medoidPath = medoidPath
    }

    enum CodingKeys: String, CodingKey {
        case clusterId = "cluster_id"
        case clusterName = "cluster_name"
        case businessFamily = "business_family"
        case businessSubmodule = "business_submodule"
        case businessDetail = "business_detail"
        case namingConfidence = "naming_confidence"
        case namingSource = "naming_source"
        case needsReview = "needs_review"
        case journeyCount = "journey_count"
        case journeyShare = "journey_share"
        case medoidPath = "medoid_path"
    }
}

/// Calibrated anomaly cutoffs and friction percentiles from thresholds.json.
public struct ScoreThresholds: Codable, Sendable, Equatable {
    public let distanceP95: Double
    public let markovP05: Double
    public let markovP01: Double
    public let backRateP90: Double
    public let loopsP90: Double
    public let revisitP90: Double
    public let spanP95: Double

    public init(
        distanceP95: Double = 0.70734,
        markovP05: Double = -4.49426,
        markovP01: Double = -5.51838,
        backRateP90: Double = 0.20,
        loopsP90: Double = 0.0,
        revisitP90: Double = 0.2609,
        spanP95: Double = 138.0
    ) {
        self.distanceP95 = distanceP95
        self.markovP05 = markovP05
        self.markovP01 = markovP01
        self.backRateP90 = backRateP90
        self.loopsP90 = loopsP90
        self.revisitP90 = revisitP90
        self.spanP95 = spanP95
    }

    enum CodingKeys: String, CodingKey {
        case distanceP95 = "distance_p95"
        case markovP05 = "markov_p05"
        case markovP01 = "markov_p01"
        case backRateP90 = "back_rate_p90"
        case loopsP90 = "loops_p90"
        case revisitP90 = "revisit_p90"
        case spanP95 = "span_p95"
    }
}

/// Markov next action candidate prediction.
public struct NextActionPrediction: Codable, Sendable, Equatable {
    public let token: String
    public let smoothedProbability: Double
    public let observedShare: Double

    public init(token: String, smoothedProbability: Double, observedShare: Double) {
        self.token = token
        self.smoothedProbability = smoothedProbability
        self.observedShare = observedShare
    }

    enum CodingKeys: String, CodingKey {
        case token
        case smoothedProbability = "smoothed_probability"
        case observedShare = "observed_share"
    }
}

// MARK: - 4. Prediction & Output Models

/// Complete model prediction result for a single journey.
public struct MobilePrediction: Codable, Sendable, Equatable {
    public let journeyId: String
    public let sessionId: String
    public let deviceId: String
    public let customerId: String?
    public let platform: String
    public let state: String
    public let boundaryReason: String
    public let eventsSeen: Int
    public let sequence: String
    public let eventSequence: [String]
    public let cluster: Int64?
    public let clusterName: String?
    public let businessFamily: String?
    public let businessSubmodule: String?
    public let businessDetail: String?
    public let namingConfidence: String?
    public let distanceToCentroid: Double?
    public let distanceLimit: Double?
    public let markovLogprob: Double?
    public let geometricAnomaly: Bool
    public let generativeAnomaly: Bool
    public let severeAnomaly: Bool
    public let frictionFlags: String
    public let nextAction: String?
    public let nextActionShare: Double?
    public let medoidPath: String?

    public init(
        journeyId: String,
        sessionId: String,
        deviceId: String = "",
        customerId: String? = nil,
        platform: String = "ios",
        state: String,
        boundaryReason: String,
        eventsSeen: Int,
        eventSequence: [String],
        cluster: Int64?,
        clusterName: String?,
        businessFamily: String? = nil,
        businessSubmodule: String? = nil,
        businessDetail: String? = nil,
        namingConfidence: String? = nil,
        distanceToCentroid: Double?,
        distanceLimit: Double? = nil,
        markovLogprob: Double?,
        geometricAnomaly: Bool = false,
        generativeAnomaly: Bool = false,
        severeAnomaly: Bool = false,
        frictionFlags: String = "",
        nextAction: String? = nil,
        nextActionShare: Double? = nil,
        medoidPath: String? = nil
    ) {
        self.journeyId = journeyId
        self.sessionId = sessionId
        self.deviceId = deviceId
        self.customerId = customerId
        self.platform = platform
        self.state = state
        self.boundaryReason = boundaryReason
        self.eventsSeen = eventsSeen
        self.sequence = eventSequence.joined(separator: " -> ")
        self.eventSequence = eventSequence
        self.cluster = cluster
        self.clusterName = clusterName
        self.businessFamily = businessFamily
        self.businessSubmodule = businessSubmodule
        self.businessDetail = businessDetail
        self.namingConfidence = namingConfidence
        self.distanceToCentroid = distanceToCentroid.map { (round($0 * 10000.0) / 10000.0) }
        self.distanceLimit = distanceLimit.map { (round($0 * 10000.0) / 10000.0) }
        self.markovLogprob = markovLogprob.map { (round($0 * 10000.0) / 10000.0) }
        self.geometricAnomaly = geometricAnomaly
        self.generativeAnomaly = generativeAnomaly
        self.severeAnomaly = severeAnomaly
        self.frictionFlags = frictionFlags
        self.nextAction = nextAction
        self.nextActionShare = nextActionShare.map { (round($0 * 10000.0) / 10000.0) }
        self.medoidPath = medoidPath
    }

    enum CodingKeys: String, CodingKey {
        case journeyId = "journey_id"
        case sessionId = "session_id"
        case deviceId = "device_id"
        case customerId = "customer_id"
        case platform
        case state
        case boundaryReason = "boundary_reason"
        case eventsSeen = "events_seen"
        case sequence
        case eventSequence = "event_sequence"
        case cluster
        case clusterName = "cluster_name"
        case businessFamily = "business_family"
        case businessSubmodule = "business_submodule"
        case businessDetail = "business_detail"
        case namingConfidence = "naming_confidence"
        case distanceToCentroid = "distance_to_centroid"
        case distanceLimit = "distance_limit"
        case markovLogprob = "markov_logprob"
        case geometricAnomaly = "geometric_anomaly"
        case generativeAnomaly = "generative_anomaly"
        case severeAnomaly = "severe_anomaly"
        case frictionFlags = "friction_flags"
        case nextAction = "next_action"
        case nextActionShare = "next_action_share"
        case medoidPath = "medoid_path"
    }

    public func toDictionary() -> [String: Any?] {
        return [
            "journey_id": journeyId,
            "session_id": sessionId,
            "device_id": deviceId.isEmpty ? nil : deviceId,
            "customer_id": customerId,
            "platform": platform,
            "state": state,
            "boundary_reason": boundaryReason,
            "events_seen": eventsSeen,
            "sequence": sequence,
            "event_sequence": eventSequence,
            "cluster": cluster,
            "cluster_name": clusterName,
            "business_family": businessFamily,
            "business_submodule": businessSubmodule,
            "business_detail": businessDetail,
            "naming_confidence": namingConfidence,
            "distance_to_centroid": distanceToCentroid,
            "distance_limit": distanceLimit,
            "markov_logprob": markovLogprob,
            "geometric_anomaly": geometricAnomaly,
            "generative_anomaly": generativeAnomaly,
            "severe_anomaly": severeAnomaly,
            "friction_flags": frictionFlags,
            "next_action": nextAction,
            "next_action_share": nextActionShare,
            "medoid_path": medoidPath
        ]
    }
}

/// Streaming processing route.
public enum MobileProcessingRoute: String, Codable, Sendable {
    case fixedDuration = "fixed_duration"
    case journeyComplete = "journey_complete"

    public var wireName: String { rawValue }
    public var displayName: String {
        switch self {
        case .fixedDuration:
            return "Fixed duration · xử lý mỗi khoảng"
        case .journeyComplete:
            return "Journey complete · gom cụm ngay khi hoàn tất"
        }
    }
}

/// Incremental update emitted during streaming replay or live execution.
public struct MobileProcessingUpdate: Codable, Sendable {
    public let route: MobileProcessingRoute
    public let windowIndex: Int?
    public let eventsInWindow: Int
    public let windowStartTimestamp: Int64?
    public let windowEndTimestamp: Int64?
    public let finalized: [MobilePrediction]
    public let provisional: [MobilePrediction]
    public let errors: [String]

    public var emitted: Bool {
        return !finalized.isEmpty || !provisional.isEmpty || !errors.isEmpty
    }

    public init(
        route: MobileProcessingRoute,
        windowIndex: Int? = nil,
        eventsInWindow: Int = 0,
        windowStartTimestamp: Int64? = nil,
        windowEndTimestamp: Int64? = nil,
        finalized: [MobilePrediction] = [],
        provisional: [MobilePrediction] = [],
        errors: [String] = []
    ) {
        self.route = route
        self.windowIndex = windowIndex
        self.eventsInWindow = eventsInWindow
        self.windowStartTimestamp = windowStartTimestamp
        self.windowEndTimestamp = windowEndTimestamp
        self.finalized = finalized
        self.provisional = provisional
        self.errors = errors
    }

    enum CodingKeys: String, CodingKey {
        case route
        case windowIndex = "window_index"
        case eventsInWindow = "events_in_window"
        case windowStartTimestamp = "window_start_timestamp"
        case windowEndTimestamp = "window_end_timestamp"
        case finalized
        case provisional
        case errors
    }
}

/// Stream update returned by session buffer appends.
public struct StreamUpdate: Sendable {
    public let finalized: [MobilePrediction]
    public let provisional: MobilePrediction?
    public let errors: [String]

    public init(finalized: [MobilePrediction], provisional: MobilePrediction?, errors: [String] = []) {
        self.finalized = finalized
        self.provisional = provisional
        self.errors = errors
    }
}

public struct MobileAnalysisWindow: Codable, Sendable {
    public let windowIndex: Int
    public let startTimestamp: Int64
    public let endTimestamp: Int64
    public let finalized: [MobilePrediction]
    public let provisional: MobilePrediction?

    public init(
        windowIndex: Int,
        startTimestamp: Int64,
        endTimestamp: Int64,
        finalized: [MobilePrediction],
        provisional: MobilePrediction?
    ) {
        self.windowIndex = windowIndex
        self.startTimestamp = startTimestamp
        self.endTimestamp = endTimestamp
        self.finalized = finalized
        self.provisional = provisional
    }

    enum CodingKeys: String, CodingKey {
        case windowIndex = "window_index"
        case startTimestamp = "start_timestamp"
        case endTimestamp = "end_timestamp"
        case finalized
        case provisional
    }
}

public struct WindowedAnalysisResult: Codable, Sendable {
    public let schemaVersion: String
    public let windowSeconds: Int
    public let windows: [MobileAnalysisWindow]

    public init(schemaVersion: String = "2.1.0", windowSeconds: Int = 30, windows: [MobileAnalysisWindow]) {
        self.schemaVersion = schemaVersion
        self.windowSeconds = windowSeconds
        self.windows = windows
    }

    enum CodingKeys: String, CodingKey {
        case schemaVersion = "schema_version"
        case windowSeconds = "window_seconds"
        case windows
    }
}

public struct MobileAnalysisResult: Codable, Sendable {
    public let schemaVersion: String
    public let predictions: [MobilePrediction]

    public init(schemaVersion: String = "2.1.0", predictions: [MobilePrediction]) {
        self.schemaVersion = schemaVersion
        self.predictions = predictions
    }

    enum CodingKeys: String, CodingKey {
        case schemaVersion = "schema_version"
        case predictions
    }
}

public struct ReplaySession: Sendable {
    public let id: String
    public let events: [ClickstreamEvent]

    public var firstEvent: ClickstreamEvent? { events.first }
    public var deviceId: String { firstEvent?.deviceId ?? "" }
    public var platform: String {
        let platforms = events.map { $0.segment }.filter { !$0.isEmpty }
        return Array(Set(platforms)).sorted().joined(separator: "/")
    }

    public init(id: String, events: [ClickstreamEvent]) {
        self.id = id
        self.events = events
    }
}
