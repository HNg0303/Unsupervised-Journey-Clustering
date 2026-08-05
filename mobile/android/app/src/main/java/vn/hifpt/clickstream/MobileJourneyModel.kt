package vn.hifpt.clickstream

import android.content.Context
import org.json.JSONArray
import org.json.JSONObject
import java.io.InputStream

data class MobilePrediction(
    val journeyId: String,
    val sessionId: String,
    val state: String,
    val boundaryReason: String,
    val eventsSeen: Int,
    val cluster: Long?,
    val classGroupCode: String?,
    val classGroup: String?,
    val classCode: String?,
    val className: String?,
    val distanceToCentroid: Double?,
    val markovLogprob: Double?,
    val geometricAnomaly: Boolean,
    val generativeAnomaly: Boolean,
    val severeAnomaly: Boolean,
    val frictionFlags: String,
    val nextAction: String?,
    val nextActionShare: Double?
) {
    fun toJson(): JSONObject = JSONObject().apply {
        put("journey_id", journeyId)
        put("session_id", sessionId)
        put("state", state)
        put("boundary_reason", boundaryReason)
        put("events_seen", eventsSeen)
        putNullable("cluster", cluster)
        putNullable("class_group_code", classGroupCode)
        putNullable("class_group", classGroup)
        putNullable("class_code", classCode)
        putNullable("class_name", className)
        putNullable("distance_to_centroid", distanceToCentroid)
        putNullable("markov_logprob", markovLogprob)
        put("geometric_anomaly", geometricAnomaly)
        put("generative_anomaly", generativeAnomaly)
        put("severe_anomaly", severeAnomaly)
        put("friction_flags", frictionFlags)
        putNullable("next_action", nextAction)
        putNullable("next_action_share", nextActionShare)
    }
}

data class MobileAnalysisResult(val predictions: List<MobilePrediction>) {
    fun toJson(): JSONObject = JSONObject().apply {
        put("schema_version", "1.0")
        put("predictions", JSONArray().apply { predictions.forEach { put(it.toJson()) } })
    }
}

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
        put("schema_version", "1.0")
        put("window_seconds", 30)
        put("windows", JSONArray().apply { windows.forEach { put(it.toJson()) } })
    }
}

/** Complete on-device scorer: raw event JSON -> journey predictions. */
class MobileJourneyModel(context: Context) : AutoCloseable {
    private val applicationContext = context.applicationContext
    private val classifier: JourneyOnnxClassifier
    private val markov: MobileMarkovBank
    private val config: MobileFrictionConfig

    init {
        val assets = context.assets
        val modelBytes = assets.open("journey_classifier.onnx").use { it.readBytes() }
        val preprocessing = assets.open("preprocessing.json").bufferedReader().use { it.readText() }
        val mapping = assets.open("class_mapping.json").bufferedReader().use { it.readText() }
        val markovJson = assets.open("markov.json").bufferedReader().use { it.readText() }
        val frictionJson = assets.open("friction_config.json").bufferedReader().use { it.readText() }
        classifier = JourneyOnnxClassifier(modelBytes, preprocessing, mapping)
        markov = MobileMarkovBank(markovJson)
        config = MobileFrictionConfig(frictionJson)
    }

    fun analyzeJson(json: String): MobileAnalysisResult =
        analyzeEvents(parseClickstreamJson(json))

    fun analyzeStream(input: InputStream): MobileAnalysisResult =
        input.bufferedReader().use { analyzeJson(it.readText()) }

    fun analyzeAsset(assetName: String = "test_data_july.json"): MobileAnalysisResult {
        val json = applicationContext.assets.open(assetName).bufferedReader().use { it.readText() }
        return analyzeJson(json)
    }

    fun analyzeEvents(events: List<ClickstreamEvent>): MobileAnalysisResult {
        val predictions = MobileJourneyPreprocessor.prepareAll(events).map {
            scorePrepared(it, provisional = false)
        }
        return MobileAnalysisResult(predictions)
    }

    /** Analyze transport windows without resetting the open journey at 30 seconds. */
    fun analyzeEventsInWindows(
        events: List<ClickstreamEvent>,
        windowSeconds: Long = 30L
    ): WindowedAnalysisResult {
        if (events.isEmpty()) return WindowedAnalysisResult(emptyList())
        val ordered = events.sortedWith(compareBy<ClickstreamEvent> { it.timestamp }.thenBy { it.sourceIndex })
        val stream = MobileJourneyStream(this)
        val windows = mutableListOf<MobileAnalysisWindow>()
        var windowStart = ordered.first().timestamp
        var windowEnd = windowStart
        var windowIndex = 1
        val finalizedInWindow = mutableListOf<MobilePrediction>()

        for (event in ordered) {
            windowEnd = event.timestamp
            val update = stream.append(event, scoreProvisional = false)
            finalizedInWindow += update.finalized
            if (event.timestamp - windowStart >= windowSeconds * 1000L) {
                windows += MobileAnalysisWindow(
                    windowIndex = windowIndex++,
                    startTimestamp = windowStart,
                    endTimestamp = event.timestamp,
                    finalized = finalizedInWindow.toList(),
                    provisional = stream.snapshotProvisional()
                )
                finalizedInWindow.clear()
                windowStart = event.timestamp
            }
        }

        val finalPrediction = stream.flush()
        if (finalPrediction != null) finalizedInWindow += finalPrediction
        windows += MobileAnalysisWindow(
            windowIndex = windowIndex,
            startTimestamp = windowStart,
            endTimestamp = windowEnd,
            finalized = finalizedInWindow.toList(),
            provisional = null
        )
        return WindowedAnalysisResult(windows)
    }

    fun scoreEvents(
        journeyId: String,
        events: List<ClickstreamEvent>,
        provisional: Boolean,
        boundaryReason: String = ""
    ): MobilePrediction? {
        val prepared = MobileJourneyPreprocessor.prepareSingle(journeyId, events, boundaryReason)
            ?: return null
        return scorePrepared(prepared, provisional)
    }

    private fun scorePrepared(prepared: PreparedJourney, provisional: Boolean): MobilePrediction {
        val state = if (prepared.nEventsFinal < config.minJourneyLength) "collecting"
        else if (provisional) "provisional" else "final"
        if (prepared.nEventsFinal < config.minJourneyLength) {
            return MobilePrediction(
                journeyId = prepared.journeyId,
                sessionId = prepared.sessionId,
                state = state,
                boundaryReason = prepared.boundaryReason,
                eventsSeen = prepared.rawEvents.size,
                cluster = null,
                classGroupCode = null,
                classGroup = null,
                classCode = null,
                className = null,
                distanceToCentroid = null,
                markovLogprob = null,
                geometricAnomaly = false,
                generativeAnomaly = false,
                severeAnomaly = false,
                frictionFlags = "",
                nextAction = null,
                nextActionShare = null
            )
        }

        val onnx = classifier.predict(JourneyFeatures(prepared.tokens, prepared.numeric))
        val logprob = markov.score(prepared.tokens, onnx.cluster)
        val generativeAnomaly = logprob != null && logprob < config.markovP05
        val severeAnomaly = onnx.geometricAnomaly && logprob != null && logprob < config.markovP01
        val flags = buildList {
            if (prepared.backRate > config.backRateP90) add("excessive_back")
            if (prepared.nLoopRemoved > config.loopsP90) add("navigation_loop")
            if (prepared.revisitRatio > config.revisitP90) add("screen_thrash")
            if (prepared.spanSeconds > config.spanP95) add("slow_journey")
            if (onnx.geometricAnomaly) add("unknown_archetype")
            if (generativeAnomaly) add("improbable_transitions")
        }.joinToString("|")
        val next = markov.predictNext(prepared.tokens, onnx.cluster)

        return MobilePrediction(
            journeyId = prepared.journeyId,
            sessionId = prepared.sessionId,
            state = state,
            boundaryReason = prepared.boundaryReason,
            eventsSeen = prepared.rawEvents.size,
            cluster = onnx.cluster,
            classGroupCode = onnx.classGroupCode,
            classGroup = onnx.classGroup,
            classCode = onnx.classCode,
            className = onnx.className,
            distanceToCentroid = onnx.distance.toDouble(),
            markovLogprob = logprob,
            geometricAnomaly = onnx.geometricAnomaly,
            generativeAnomaly = generativeAnomaly,
            severeAnomaly = severeAnomaly,
            frictionFlags = flags,
            nextAction = next?.token,
            nextActionShare = next?.observedShare
        )
    }

    override fun close() = classifier.close()

    private class MobileFrictionConfig(json: String) {
        private val value = JSONObject(json)
        val markovP05 = value.optDouble("markov_p05", -4.6398522796713)
        val markovP01 = value.optDouble("markov_p01", -5.03812319276212)
        val backRateP90 = value.optDouble("back_rate_p90", 0.2)
        val loopsP90 = value.optInt("loops_p90", 0)
        val revisitP90 = value.optDouble("revisit_p90", 0.3333)
        val spanP95 = value.optDouble("span_p95", 104.1425)
        val minJourneyLength = value.optInt("min_journey_length", 4)
    }
}

private fun JSONObject.putNullable(key: String, value: Any?) {
    put(key, value ?: JSONObject.NULL)
}
