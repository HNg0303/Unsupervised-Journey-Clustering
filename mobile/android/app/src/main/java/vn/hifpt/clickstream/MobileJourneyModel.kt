package vn.hifpt.clickstream

import android.content.Context
import android.content.res.AssetManager
import org.json.JSONArray
import org.json.JSONObject
import java.io.InputStream

data class MobilePrediction(
    val journeyId: String,
    val sessionId: String,
    val state: String,
    val boundaryReason: String,
    val eventsSeen: Int,
    val eventSequence: List<String>,
    val cluster: Long?,
    val clusterNamespace: String?,
    val effectiveClusterKey: String?,
    val assignmentType: String?,
    val classGroupCode: String?,
    val classGroup: String?,
    val classCode: String?,
    val className: String?,
    val clusterName: String?,
    val clusterNameEn: String?,
    val distanceToCentroid: Double?,
    val markovLogprob: Double?,
    val primaryCluster: Long?,
    val primaryDistance: Double?,
    val primaryDistanceLimit: Double?,
    val secondaryCluster: Long?,
    val secondaryDistance: Double?,
    val secondaryDistanceLimit: Double?,
    val secondaryDistanceMargin: Double?,
    val secondaryMarkovLimit: Double?,
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
        put("sequence", eventSequence.joinToString(" -> "))
        put("event_sequence", JSONArray().apply { eventSequence.forEach { put(it) } })
        putNullable("cluster", cluster)
        putNullable("cluster_namespace", clusterNamespace)
        putNullable("effective_cluster_key", effectiveClusterKey)
        putNullable("assignment_type", assignmentType)
        putNullable("class_group_code", classGroupCode)
        putNullable("class_group", classGroup)
        putNullable("class_code", classCode)
        putNullable("class_name", className)
        putNullable("cluster_name", clusterName)
        putNullable("cluster_name_en", clusterNameEn)
        putNullable("distance_to_centroid", distanceToCentroid)
        putNullable("markov_logprob", markovLogprob)
        putNullable("primary_cluster", primaryCluster)
        putNullable("primary_distance", primaryDistance)
        putNullable("primary_distance_limit", primaryDistanceLimit)
        putNullable("secondary_cluster", secondaryCluster)
        putNullable("secondary_distance", secondaryDistance)
        putNullable("secondary_distance_limit", secondaryDistanceLimit)
        putNullable("secondary_distance_margin", secondaryDistanceMargin)
        putNullable("secondary_markov_limit", secondaryMarkovLimit)
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
        put("schema_version", "3.0")
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
        put("schema_version", "3.0")
        put("window_seconds", 30)
        put("windows", JSONArray().apply { windows.forEach { put(it.toJson()) } })
    }
}

/** Complete on-device scorer: raw event JSON -> journey predictions. */
class MobileJourneyModel(context: Context) : AutoCloseable {
    private val applicationContext = context.applicationContext
    private val primaryClassifier: JourneyOnnxClassifier
    private val assets: AssetManager = context.assets
    @Volatile private var secondaryClassifier: JourneyOnnxClassifier? = null
    private val primaryMarkov: MobileMarkovBank
    @Volatile private var secondaryMarkov: MobileMarkovBank? = null
    private val hierarchy: HierarchicalConfig
    private val config: MobileFrictionConfig

    init {
        val frictionJson = assets.readText("friction_config.json")
        // The segmentation contract is tiny in friction_config. Avoid parsing the
        // 33 MB preprocessing JSON a second time merely to read three settings.
        MobileJourneyPreprocessor.configure(frictionJson)
        primaryClassifier = createClassifier("c")
        primaryMarkov = MobileMarkovBank(assets.readText("c_markov.json"))
        hierarchy = HierarchicalConfig(assets.readText("hierarchical_config.json"))
        config = MobileFrictionConfig(frictionJson)
    }

    fun analyzeJson(json: String): MobileAnalysisResult =
        analyzeEvents(parseClickstreamJson(json))

    fun analyzeStream(input: InputStream): MobileAnalysisResult =
        input.bufferedReader().use { analyzeJson(it.readText()) }

    fun analyzeAsset(assetName: String = "test_data_android.csv"): MobileAnalysisResult {
        val source = applicationContext.assets.open(assetName).bufferedReader().use { it.readText() }
        val events = if (assetName.lowercase(java.util.Locale.US).endsWith(".csv")) {
            ClickstreamRepository(applicationContext).parseCsvEvents(source)
        } else {
            parseClickstreamJson(source)
        }
        return analyzeEvents(events)
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
                eventSequence = prepared.tokens,
                cluster = null,
                clusterNamespace = null,
                effectiveClusterKey = null,
                assignmentType = null,
                classGroupCode = null,
                classGroup = null,
                classCode = null,
                className = null,
                clusterName = null,
                clusterNameEn = null,
                distanceToCentroid = null,
                markovLogprob = null,
                primaryCluster = null,
                primaryDistance = null,
                primaryDistanceLimit = null,
                secondaryCluster = null,
                secondaryDistance = null,
                secondaryDistanceLimit = null,
                secondaryDistanceMargin = null,
                secondaryMarkovLimit = null,
                geometricAnomaly = false,
                generativeAnomaly = false,
                severeAnomaly = false,
                frictionFlags = "",
                nextAction = null,
                nextActionShare = null
            )
        }

        val features = JourneyFeatures(prepared.tokens, prepared.numeric)
        val primary = primaryClassifier.predict(features)
        val primaryLimit = hierarchy.cDistanceLimits[primary.cluster]
        val primaryAccepted = primaryLimit != null && primary.distance <= primaryLimit
        val secondaryRuntime = if (primaryAccepted) null else getSecondaryRuntime()
        val secondary = secondaryRuntime?.classifier?.predict(features)
        val secondaryLimit = secondary?.cluster?.let { hierarchy.bDistanceLimits[it] }
        val secondaryMarkovLimit = secondary?.cluster?.let { hierarchy.bMarkovLimits[it] }
        val secondaryLogprob = secondary?.let { secondaryRuntime!!.markov.score(prepared.tokens, it.cluster) }
        val secondaryMargin = secondary?.secondDistance?.let {
            (it - secondary.distance) / secondary.distance.coerceAtLeast(1e-12f)
        }
        val secondaryAccepted = secondary != null && secondaryLimit != null &&
            secondary.distance <= secondaryLimit && secondaryMarkovLimit != null &&
            secondaryLogprob != null && secondaryLogprob >= secondaryMarkovLimit &&
            secondaryMargin != null && secondaryMargin >= hierarchy.minDistanceMargin
        val chosen = if (primaryAccepted) primary else if (secondaryAccepted) secondary else null
        val namespace = if (primaryAccepted) "C" else if (secondaryAccepted) "B" else null
        val effectiveKey = chosen?.let { "$namespace:${it.cluster}" } ?: "UNKNOWN"
        val assignmentType = when {
            primaryAccepted -> "C_primary"
            secondaryAccepted -> "B_noise_fallback"
            else -> "unresolved_noise"
        }
        val logprob = when (namespace) {
            "C" -> primaryMarkov.score(prepared.tokens, primary.cluster)
            "B" -> secondaryLogprob
            else -> null
        }
        val geometricAnomaly = chosen == null
        val generativeAnomaly = logprob != null && logprob < config.markovP05
        val severeAnomaly = geometricAnomaly && logprob != null && logprob < config.markovP01
        val flags = buildList {
            if (prepared.backRate > config.backRateP90) add("excessive_back")
            if (prepared.nLoopRemoved > config.loopsP90) add("navigation_loop")
            if (prepared.revisitRatio > config.revisitP90) add("screen_thrash")
            if (prepared.spanSeconds > config.spanP95) add("slow_journey")
            if (geometricAnomaly) add("unknown_archetype")
            if (generativeAnomaly) add("improbable_transitions")
        }.joinToString("|")
        val next = when (namespace) {
            "C" -> primaryMarkov.predictNext(prepared.tokens, primary.cluster)
            "B" -> secondaryRuntime!!.markov.predictNext(prepared.tokens, secondary!!.cluster)
            else -> null
        }

        return MobilePrediction(
            journeyId = prepared.journeyId,
            sessionId = prepared.sessionId,
            state = state,
            boundaryReason = prepared.boundaryReason,
            eventsSeen = prepared.rawEvents.size,
            eventSequence = prepared.tokens,
            cluster = chosen?.cluster,
            clusterNamespace = namespace,
            effectiveClusterKey = effectiveKey,
            assignmentType = assignmentType,
            classGroupCode = chosen?.classGroupCode ?: "unknown",
            classGroup = chosen?.classGroup ?: "chưa phân loại",
            classCode = chosen?.classCode ?: "unknown_journey",
            className = chosen?.className ?: "Hành trình chưa phân loại / hỗn hợp",
            clusterName = chosen?.clusterName ?: "Hành trình chưa phân loại / hỗn hợp",
            clusterNameEn = chosen?.clusterNameEn ?: "Unclassified / mixed journeys",
            distanceToCentroid = chosen?.distance?.toDouble(),
            markovLogprob = logprob,
            primaryCluster = primary.cluster,
            primaryDistance = primary.distance.toDouble(),
            primaryDistanceLimit = primaryLimit,
            secondaryCluster = secondary?.cluster,
            secondaryDistance = secondary?.distance?.toDouble(),
            secondaryDistanceLimit = secondaryLimit,
            secondaryDistanceMargin = secondaryMargin?.toDouble(),
            secondaryMarkovLimit = secondaryMarkovLimit,
            geometricAnomaly = geometricAnomaly,
            generativeAnomaly = generativeAnomaly,
            severeAnomaly = severeAnomaly,
            frictionFlags = flags,
            nextAction = next?.token,
            nextActionShare = next?.observedShare
        )
    }

    override fun close() {
        primaryClassifier.close()
        secondaryClassifier?.close()
    }

    @Synchronized
    private fun getSecondaryRuntime(): SecondaryRuntime {
        val existingClassifier = secondaryClassifier
        val existingMarkov = secondaryMarkov
        if (existingClassifier != null && existingMarkov != null) {
            return SecondaryRuntime(existingClassifier, existingMarkov)
        }
        val classifier = createClassifier("b")
        val markov = MobileMarkovBank(assets.readText("b_markov.json"))
        secondaryClassifier = classifier
        secondaryMarkov = markov
        return SecondaryRuntime(classifier, markov)
    }

    private data class SecondaryRuntime(
        val classifier: JourneyOnnxClassifier,
        val markov: MobileMarkovBank
    )

    private fun createClassifier(prefix: String): JourneyOnnxClassifier {
        val preprocessing = assets.readText("${prefix}_preprocessing.json")
        return JourneyOnnxClassifier(
            assets.readBytes("${prefix}_journey_classifier.onnx"), preprocessing,
            assets.readText("${prefix}_class_mapping.json")
        )
    }

    private class HierarchicalConfig(json: String) {
        private val value = JSONObject(json)
        val minDistanceMargin = value.getDouble("min_distance_margin")
        val cDistanceLimits = value.getJSONObject("c_distance_limits").toLongDoubleMap()
        val bDistanceLimits = value.getJSONObject("b_distance_limits").toLongDoubleMap()
        val bMarkovLimits = value.getJSONObject("b_markov_limits").toLongDoubleMap()
    }

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

private fun android.content.res.AssetManager.readText(name: String): String =
    open(name).bufferedReader().use { it.readText() }

private fun android.content.res.AssetManager.readBytes(name: String): ByteArray =
    open(name).use { it.readBytes() }

private fun JSONObject.toLongDoubleMap(): Map<Long, Double> =
    keys().asSequence().associate { it.toLong() to getDouble(it) }
