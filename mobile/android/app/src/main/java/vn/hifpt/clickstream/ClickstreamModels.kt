package vn.hifpt.clickstream

import org.json.JSONArray
import org.json.JSONObject

// --------------------------------------------------------------------------
// 1. Event Models
// --------------------------------------------------------------------------

/** Raw clickstream event parsed from production CSV or real-time SDK JSON stream. */
data class ClickstreamEvent(
    val sourceIndex: Int,
    val eventId: String,
    val deviceId: String,
    val customerId: String?,
    val sessionId: String,
    val createdAt: String?,
    val timestamp: Long,
    val key: String,
    val segment: String,
    val name: String,
    val screenId: String = "",
    val durationSeconds: Int = 0,
    val visit: String? = null
)

/** Normalized click with canonical paths, resolved screen context, and semantic tokens. */
data class CanonicalClick(
    val raw: ClickstreamEvent,
    val eventType: String,
    val segmentName: String,
    val screenContext: String,
    val eventToken: String,
    val eventTimeMillis: Long,
    val gapPrevSeconds: Double?,
    val gapNextSeconds: Double?,
    val semantics: SemanticLabel = SemanticLabel(),
    val tokenL1: String = "",
    val tokenL2: String = "",
    val tokenL3: String = "",
    val operationToken: String = ""
)

/** Coarse business semantics resolved from the path taxonomy. */
data class SemanticLabel(
    val businessFamily: String = "unknown",
    val businessModule: String = "unknown",
    val businessObject: String = "unknown",
    val operation: String = "unknown",
    val operationStage: String = "unknown",
    val semanticConfidence: Double = 0.0,
    val semanticEvidence: String = ""
) {
    val isKnown: Boolean get() = businessFamily != "unknown"
}

// --------------------------------------------------------------------------
// 2. Journey Data Models
// --------------------------------------------------------------------------

/** Segmented and cleaned journey sequence ready for feature extraction and scoring. */
data class PreparedJourney(
    val journeyId: String,
    val sessionId: String,
    val deviceId: String,
    val customerId: String?,
    val platform: String,
    val boundaryReason: String,
    val rawEvents: List<ClickstreamEvent>,
    val cleanedEvents: List<CanonicalClick>,
    val tokens: List<String>,
    val channels: Map<String, List<String>>,
    val numeric: Map<String, Double>,
    val nEventsFinal: Int,
    val nLoopRemoved: Int,
    val nDedupRemoved: Int,
    val backRate: Double,
    val revisitRatio: Double,
    val spanSeconds: Double
)

/** Input feature bundle for JourneyScorer / JourneyVectorizer. */
data class JourneyFeatures(
    val tokens: List<String>,
    val numeric: Map<String, Double>,
    val channels: Map<String, List<String>> = emptyMap()
)

/** Multi-channel feature matrices and final projected embedding. */
data class JourneyFeatureMatrix(
    val primaryEmbedding: FloatArray,
    val channelEmbeddings: Map<String, FloatArray>,
    val scaledNumericBlock: FloatArray,
    val concatenatedVector: FloatArray,
    val projectedEmbedding: FloatArray
)

// --------------------------------------------------------------------------
// 3. Archetype & Threshold Models
// --------------------------------------------------------------------------

/** Business mapping metadata for an archetype cluster from cluster_mapping.json. */
data class ClusterArchetype(
    val clusterId: Long,
    val clusterName: String,
    val businessFamily: String = "Chưa phân loại",
    val businessSubmodule: String = "",
    val businessDetail: String = "",
    val namingConfidence: String = "medium",
    val namingSource: String = "",
    val needsReview: Boolean = false,
    val journeyCount: Int = 0,
    val journeyShare: Double = 0.0,
    val medoidPath: String = ""
)

/** Calibrated anomaly cutoffs and friction percentiles from thresholds.json. */
data class ScoreThresholds(
    val distanceP95: Double = 0.70734,
    val markovP05: Double = -4.49426,
    val markovP01: Double = -5.51838,
    val backRateP90: Double = 0.20,
    val loopsP90: Double = 0.0,
    val revisitP90: Double = 0.2609,
    val spanP95: Double = 138.0
)

/** Markov next action candidate prediction. */
data class NextActionPrediction(
    val token: String,
    val smoothedProbability: Double,
    val observedShare: Double
)

// --------------------------------------------------------------------------
// 4. Prediction & Output Models
// --------------------------------------------------------------------------

/** Complete model prediction result for a single journey. */
data class MobilePrediction(
    val journeyId: String,
    val sessionId: String,
    val deviceId: String = "",
    val customerId: String? = null,
    val platform: String = "android",
    val state: String,
    val boundaryReason: String,
    val eventsSeen: Int,
    val eventSequence: List<String>,
    val cluster: Long?,
    val clusterName: String?,
    val businessFamily: String? = null,
    val businessSubmodule: String? = null,
    val businessDetail: String? = null,
    val namingConfidence: String? = null,
    val distanceToCentroid: Double?,
    val distanceLimit: Double? = null,
    val markovLogprob: Double?,
    val geometricAnomaly: Boolean = false,
    val generativeAnomaly: Boolean = false,
    val severeAnomaly: Boolean = false,
    val frictionFlags: String = "",
    val nextAction: String? = null,
    val nextActionShare: Double? = null,
    val medoidPath: String? = null
) {
    fun toJson(): JSONObject = JSONObject().apply {
        put("journey_id", journeyId)
        put("session_id", sessionId)
        putNullable("device_id", deviceId.takeIf { it.isNotBlank() })
        putNullable("customer_id", customerId)
        put("platform", platform)
        put("state", state)
        put("boundary_reason", boundaryReason)
        put("events_seen", eventsSeen)
        put("sequence", eventSequence.joinToString(" -> "))
        put("event_sequence", JSONArray().apply { eventSequence.forEach { put(it) } })
        putNullable("cluster", cluster)
        putNullable("cluster_name", clusterName)
        putNullable("business_family", businessFamily)
        putNullable("business_submodule", businessSubmodule)
        putNullable("business_detail", businessDetail)
        putNullable("naming_confidence", namingConfidence)
        putNullable("distance_to_centroid", distanceToCentroid?.let { round4(it) })
        putNullable("distance_limit", distanceLimit?.let { round4(it) })
        putNullable("markov_logprob", markovLogprob?.let { round4(it) })
        put("geometric_anomaly", geometricAnomaly)
        put("generative_anomaly", generativeAnomaly)
        put("severe_anomaly", severeAnomaly)
        put("friction_flags", frictionFlags)
        putNullable("next_action", nextAction)
        putNullable("next_action_share", nextActionShare?.let { round4(it) })
        putNullable("medoid_path", medoidPath)
    }

    private fun round4(v: Double): Double = (Math.round(v * 10000.0) / 10000.0)
}

/** Streaming processing route. */
enum class MobileProcessingRoute(val wireName: String, val displayName: String) {
    FIXED_DURATION("fixed_duration", "Fixed duration · xử lý mỗi khoảng"),
    JOURNEY_COMPLETE("journey_complete", "Journey complete · gom cụm ngay khi hoàn tất")
}

/** Incremental update emitted during streaming replay or live execution. */
data class MobileProcessingUpdate(
    val route: MobileProcessingRoute,
    val windowIndex: Int? = null,
    val eventsInWindow: Int = 0,
    val windowStartTimestamp: Long? = null,
    val windowEndTimestamp: Long? = null,
    val finalized: List<MobilePrediction> = emptyList(),
    val provisional: List<MobilePrediction> = emptyList(),
    val errors: List<String> = emptyList()
) {
    val emitted: Boolean get() = finalized.isNotEmpty() || provisional.isNotEmpty() || errors.isNotEmpty()

    fun toJson(): JSONObject = JSONObject().apply {
        put("route", route.wireName)
        putNullable("window_index", windowIndex)
        put("events_in_window", eventsInWindow)
        putNullable("window_start_timestamp", windowStartTimestamp)
        putNullable("window_end_timestamp", windowEndTimestamp)
        put("finalized", JSONArray().apply { finalized.forEach { put(it.toJson()) } })
        put("provisional", JSONArray().apply { provisional.forEach { put(it.toJson()) } })
        put("errors", JSONArray().apply { errors.forEach { put(it) } })
    }
}

/** Legacy-compatible stream update wrapper. */
data class StreamUpdate(
    val finalized: List<MobilePrediction>,
    val provisional: MobilePrediction?,
    val errors: List<String> = emptyList()
)

data class MobileAnalysisWindow(
    val windowIndex: Int,
    val startTimestamp: Long,
    val endTimestamp: Long,
    val finalized: List<MobilePrediction>,
    val provisional: MobilePrediction?
) {
    fun toJson(): JSONObject = JSONObject().apply {
        put("window_index", windowIndex)
        put("start_timestamp", startTimestamp)
        put("end_timestamp", endTimestamp)
        put("finalized", JSONArray().apply { finalized.forEach { put(it.toJson()) } })
        put("provisional", provisional?.toJson() ?: JSONObject.NULL)
    }
}

data class WindowedAnalysisResult(val windows: List<MobileAnalysisWindow>) {
    fun toJson(): JSONObject = JSONObject().apply {
        put("schema_version", "2.1.0")
        put("window_seconds", 30)
        put("windows", JSONArray().apply { windows.forEach { put(it.toJson()) } })
    }
}

data class MobileAnalysisResult(val predictions: List<MobilePrediction>) {
    fun toJson(): JSONObject = JSONObject().apply {
        put("schema_version", "2.1.0")
        put("predictions", JSONArray().apply { predictions.forEach { put(it.toJson()) } })
    }
}

data class ReplaySession(
    val id: String,
    val events: List<ClickstreamEvent>
) {
    val firstEvent: ClickstreamEvent get() = events.first()
    val deviceId: String get() = firstEvent.deviceId
    val platform: String get() = events.map { it.segment }.filter { it.isNotBlank() }.distinct().joinToString("/")
}

private fun JSONObject.putNullable(key: String, value: Any?) {
    put(key, value ?: JSONObject.NULL)
}
