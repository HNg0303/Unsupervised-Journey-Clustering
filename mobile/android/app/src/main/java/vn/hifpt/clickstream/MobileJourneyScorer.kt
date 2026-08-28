package vn.hifpt.clickstream

import android.content.Context
import android.content.res.AssetManager
import org.json.JSONArray
import org.json.JSONObject
import java.io.InputStream
import java.nio.ByteBuffer
import java.nio.ByteOrder
import kotlin.math.ln
import kotlin.math.max
import kotlin.math.min
import kotlin.math.sqrt

// --------------------------------------------------------------------------
// 1. Native Float16 & Math Utilities
// --------------------------------------------------------------------------

internal object Float16Helper {
    /** Convert 16-bit IEEE 754 half-precision float to 32-bit float without native deps. */
    fun halfToFloat(hbits: Short): Float {
        val h = hbits.toInt() and 0xffff
        var mant = h and 0x03ff
        val exp = h and 0x7c00
        val sign = if (h and 0x8000 != 0) -1.0f else 1.0f

        if (exp == 0x7c00) {
            return if (mant != 0) Float.NaN else sign * Float.POSITIVE_INFINITY
        }
        if (exp != 0) {
            val fExp = (exp shr 10) - 15 + 127
            return Float.fromBits(((h and 0x8000) shl 16) or (fExp shl 23) or (mant shl 13))
        }
        if (mant == 0) return sign * 0.0f
        // Subnormal
        while ((mant and 0x0400) == 0) {
            mant = mant shl 1
        }
        mant = mant and 0x03ff.inv()
        val fExp = 1 - 15 + 127
        return Float.fromBits(((h and 0x8000) shl 16) or (fExp shl 23) or (mant shl 13))
    }

    fun l2Normalize(values: FloatArray) {
        var sumSq = 0.0
        for (v in values) sumSq += (v * v)
        val norm = sqrt(sumSq).toFloat()
        if (norm > 1e-12f) {
            for (i in values.indices) values[i] /= norm
        }
    }

    fun l2Normalize(values: DoubleArray) {
        var sumSq = 0.0
        for (v in values) sumSq += (v * v)
        val norm = sqrt(sumSq)
        if (norm > 1e-12) {
            for (i in values.indices) values[i] /= norm
        }
    }
}

// --------------------------------------------------------------------------
// 2. Journey Vectorizer (Multi-Channel SVD*IDF + Numeric + Global PCA)
// --------------------------------------------------------------------------

class MobileJourneyVectorizer(
    private val vocabularies: Map<String, Map<String, Int>>,
    private val channelWeights: Map<String, Float>,
    private val numericBlockWeight: Float,
    private val numericColumns: List<String>,
    private val numericLog1pColumns: Set<String>,
    private val numericMean: FloatArray,
    private val numericScale: FloatArray,
    private val primaryW: Array<FloatArray>, // [48, vocab_primary]
    private val channelW: Map<String, Array<FloatArray>>, // channel -> [48, vocab_ch]
    private val globalPcaComp: Array<FloatArray>?, // [48, 203]
    private val globalPcaMean: FloatArray? // [203]
) {
    val embeddingDim: Int = globalPcaComp?.size ?: 48

    fun transform(journey: PreparedJourney): JourneyFeatureMatrix {
        val primaryEmb = encodeChannel(journey.tokens, "primary")
        val channelEmbs = mutableMapOf<String, FloatArray>()
        for ((chName, tokens) in journey.channels) {
            if (chName != "primary" && channelW.containsKey(chName)) {
                channelEmbs[chName] = encodeChannel(tokens, chName)
            }
        }

        val numericScaled = encodeNumeric(journey.numeric)

        // Concatenate blocks: primary + channels + numeric
        val totalLen = 48 * (1 + channelEmbs.size) + numericColumns.size
        val concatenated = FloatArray(totalLen)
        var offset = 0

        // 1. Primary
        val wPrim = channelWeights["primary"] ?: 1.0f
        for (i in 0 until 48) concatenated[offset + i] = primaryEmb[i] * wPrim
        offset += 48

        // 2. Sorted channels
        for (chName in channelW.keys.sorted()) {
            val emb = channelEmbs[chName] ?: FloatArray(48)
            val wCh = channelWeights[chName] ?: 1.0f
            for (i in 0 until 48) concatenated[offset + i] = emb[i] * wCh
            offset += 48
        }

        // 3. Numeric block
        for (i in numericScaled.indices) {
            concatenated[offset + i] = numericScaled[i] * numericBlockWeight
        }

        // 4. Global PCA projection
        val projected: FloatArray
        if (globalPcaComp != null && globalPcaMean != null) {
            projected = FloatArray(globalPcaComp.size)
            for (compIdx in globalPcaComp.indices) {
                val compRow = globalPcaComp[compIdx]
                var dot = 0.0f
                for (i in concatenated.indices) {
                    dot += (concatenated[i] - globalPcaMean[i]) * compRow[i]
                }
                projected[compIdx] = dot
            }
            Float16Helper.l2Normalize(projected)
        } else {
            projected = concatenated.clone()
            Float16Helper.l2Normalize(projected)
        }

        return JourneyFeatureMatrix(
            primaryEmbedding = primaryEmb,
            channelEmbeddings = channelEmbs,
            scaledNumericBlock = numericScaled,
            concatenatedVector = concatenated,
            projectedEmbedding = projected
        )
    }

    private fun encodeChannel(sequence: List<String>, channelName: String): FloatArray {
        val vocab = vocabularies[channelName] ?: return FloatArray(48)
        val W = if (channelName == "primary") primaryW else channelW[channelName] ?: return FloatArray(48)

        val bounded = ArrayList<String>(sequence.size + 2)
        bounded.add("<bos>")
        bounded.addAll(sequence)
        bounded.add("<eos>")

        // Extract 1-grams and 2-grams
        val counts = mutableMapOf<String, Int>()
        for (n in 1..2) {
            for (i in 0..(bounded.size - n)) {
                val ngram = if (n == 1) bounded[i] else "${bounded[i]} ${bounded[i + 1]}"
                counts[ngram] = (counts[ngram] ?: 0) + 1
            }
        }

        val z = FloatArray(48)
        for ((ngram, cnt) in counts) {
            val idx = vocab[ngram] ?: continue
            val tf = (1.0 + ln(cnt.toDouble())).toFloat()
            for (comp in 0 until 48) {
                z[comp] += tf * W[comp][idx]
            }
        }

        Float16Helper.l2Normalize(z)
        return z
    }

    private fun encodeNumeric(metrics: Map<String, Double>): FloatArray {
        val result = FloatArray(numericColumns.size)
        for (i in numericColumns.indices) {
            val col = numericColumns[i]
            var raw = metrics[col] ?: 0.0
            if (col in numericLog1pColumns) {
                raw = ln(max(0.0, raw) + 1.0)
            }
            val scale = if (numericScale[i] == 0.0f) 1.0f else numericScale[i]
            result[i] = ((raw.toFloat() - numericMean[i]) / scale)
        }
        Float16Helper.l2Normalize(result)
        return result
    }

    companion object {
        fun load(
            vocabJson: String,
            metaJson: String,
            weightsBytes: ByteArray
        ): MobileJourneyVectorizer {
            val vocabObj = JSONObject(vocabJson)
            val vocabs = mutableMapOf<String, Map<String, Int>>()
            for (key in vocabObj.keys()) {
                val inner = vocabObj.getJSONObject(key)
                val map = mutableMapOf<String, Int>()
                for (term in inner.keys()) {
                    map[term] = inner.getInt(term)
                }
                vocabs[key] = map
            }

            val metaObj = JSONObject(metaJson)
            val cwObj = metaObj.getJSONObject("channel_weights")
            val channelWeights = mutableMapOf<String, Float>()
            for (k in cwObj.keys()) channelWeights[k] = cwObj.getDouble(k).toFloat()

            val numericBlockWeight = metaObj.optDouble("numeric_block_weight", 0.35).toFloat()
            val numericColumns = metaObj.getJSONArray("numeric_columns").toStringList()
            val numericLog1pColumns = metaObj.getJSONArray("numeric_log1p_columns").toStringList().toSet()
            val numericMean = metaObj.getJSONArray("numeric_mean").toFloatArray()
            val numericScale = metaObj.getJSONArray("numeric_scale").toFloatArray()

            val hasGlobalPca = metaObj.optBoolean("has_global_pca", false)
            val pcaDim = if (hasGlobalPca) metaObj.optInt("global_pca_components_count", 48) else 0

            val buffer = ByteBuffer.wrap(weightsBytes).order(ByteOrder.LITTLE_ENDIAN)

            // Read Primary W [48, vocab_primary] (Float16)
            val pVocabSize = vocabs["primary"]?.size ?: 20000
            val primaryW = Array(48) { FloatArray(pVocabSize) }
            for (comp in 0 until 48) {
                for (termIdx in 0 until pVocabSize) {
                    primaryW[comp][termIdx] = Float16Helper.halfToFloat(buffer.short)
                }
            }

            // Read Channel W matrices
            val channelW = mutableMapOf<String, Array<FloatArray>>()
            for (chName in vocabs.keys.sorted()) {
                if (chName == "primary") continue
                val chVocabSize = vocabs[chName]?.size ?: 0
                val chMat = Array(48) { FloatArray(chVocabSize) }
                for (comp in 0 until 48) {
                    for (termIdx in 0 until chVocabSize) {
                        chMat[comp][termIdx] = Float16Helper.halfToFloat(buffer.short)
                    }
                }
                channelW[chName] = chMat
            }

            // Read Global PCA
            val globalPcaComp: Array<FloatArray>?
            val globalPcaMean: FloatArray?
            if (hasGlobalPca && pcaDim > 0) {
                val inDim = 48 * (1 + channelW.size) + numericColumns.size
                globalPcaComp = Array(pcaDim) { FloatArray(inDim) }
                for (d in 0 until pcaDim) {
                    for (i in 0 until inDim) {
                        globalPcaComp[d][i] = Float16Helper.halfToFloat(buffer.short)
                    }
                }
                globalPcaMean = FloatArray(inDim)
                for (i in 0 until inDim) {
                    globalPcaMean[i] = buffer.float
                }
            } else {
                globalPcaComp = null
                globalPcaMean = null
            }

            return MobileJourneyVectorizer(
                vocabularies = vocabs,
                channelWeights = channelWeights,
                numericBlockWeight = numericBlockWeight,
                numericColumns = numericColumns,
                numericLog1pColumns = numericLog1pColumns,
                numericMean = numericMean,
                numericScale = numericScale,
                primaryW = primaryW,
                channelW = channelW,
                globalPcaComp = globalPcaComp,
                globalPcaMean = globalPcaMean
            )
        }
    }
}

// --------------------------------------------------------------------------
// 3. Centroid Distance & Archetype Assigner
// --------------------------------------------------------------------------

class MobileCentroidAssigner(
    val clusterIds: LongArray,
    val centroids: Array<FloatArray> // [n_clusters, 48]
) {
    fun assign(embedding: FloatArray): AssignmentResult {
        var minDistance = Float.POSITIVE_INFINITY
        var bestIndex = 0

        for (cIdx in centroids.indices) {
            val centroid = centroids[cIdx]
            var sumSq = 0.0f
            for (d in 0 until min(embedding.size, centroid.size)) {
                val diff = embedding[d] - centroid[d]
                sumSq += (diff * diff)
            }
            val dist = sqrt(sumSq)
            if (dist < minDistance) {
                minDistance = dist
                bestIndex = cIdx
            }
        }

        val bestClusterId = if (clusterIds.isNotEmpty()) clusterIds[bestIndex] else bestIndex.toLong()
        return AssignmentResult(bestClusterId, minDistance.toDouble())
    }

    data class AssignmentResult(val clusterId: Long, val distance: Double)

    companion object {
        fun load(manifestJson: String, binaryCentroids: ByteArray): MobileCentroidAssigner {
            val manifest = JSONObject(manifestJson)
            val clusterIdsArray = manifest.getJSONArray("cluster_ids")
            val nClusters = clusterIdsArray.length()
            val clusterIds = LongArray(nClusters) { clusterIdsArray.getLong(it) }
            val embeddingDim = manifest.optInt("embedding_dim", 48)

            val buffer = ByteBuffer.wrap(binaryCentroids).order(ByteOrder.LITTLE_ENDIAN)
            val centroids = Array(nClusters) { FloatArray(embeddingDim) }
            for (c in 0 until nClusters) {
                for (d in 0 until embeddingDim) {
                    centroids[c][d] = Float16Helper.halfToFloat(buffer.short)
                }
            }
            return MobileCentroidAssigner(clusterIds, centroids)
        }
    }
}

// --------------------------------------------------------------------------
// 4. Markov Companion Model
// --------------------------------------------------------------------------

class MobileMarkovBank(markovJson: String) {
    private data class Edge(val token: String, val count: Double)
    private data class Bucket(
        val transitions: Map<String, List<Edge>>,
        val totals: Map<String, Double>
    )

    private val smoothing: Double
    private val vocabSize: Int
    private val vocabulary: Set<String>
    private val clusters: Map<Long, Bucket>
    private val global: Bucket

    init {
        val root = JSONObject(markovJson)
        smoothing = root.optDouble("smoothing", 0.5)
        vocabSize = root.optInt("vocab_size", root.optJSONArray("vocabulary")?.length() ?: 1000)
        vocabulary = root.optJSONArray("vocabulary")?.toStringList()?.toSet() ?: emptySet()

        val globalObj = root.optJSONObject("global") ?: JSONObject()
        global = parseBucket(globalObj)

        val clustersObj = root.optJSONObject("clusters") ?: JSONObject()
        val cMap = mutableMapOf<Long, Bucket>()
        for (k in clustersObj.keys()) {
            val bObj = clustersObj.getJSONObject(k)
            cMap[k.toLong()] = parseBucket(bObj)
        }
        clusters = cMap
    }

    fun score(tokens: List<String>, clusterId: Long?): Double? {
        if (tokens.size < 2) return null
        val bucket = (clusterId?.let { clusters[it] }) ?: global
        val denominatorMass = smoothing * max(vocabSize, 1).toDouble()
        var totalLogprob = 0.0
        var pairs = 0

        for (i in 0 until tokens.size - 1) {
            val src = tokens[i]
            val dst = tokens[i + 1]
            if (vocabulary.isNotEmpty() && (src !in vocabulary || dst !in vocabulary)) continue

            val edges = bucket.transitions[src]
            val count = edges?.firstOrNull { it.token == dst }?.count ?: 0.0
            val total = (bucket.totals[src] ?: 0.0) + denominatorMass

            totalLogprob += (ln(count + smoothing) - ln(total))
            pairs++
        }

        return if (pairs == 0) null else totalLogprob / pairs
    }

    fun predictNext(tokens: List<String>, clusterId: Long?, topK: Int = 3): NextActionPrediction? {
        val src = tokens.lastOrNull() ?: return null
        if (vocabulary.isNotEmpty() && src !in vocabulary) return null
        val bucket = (clusterId?.let { clusters[it] }) ?: global
        val edges = bucket.transitions[src] ?: return null
        if (edges.isEmpty()) return null

        val totalObserved = edges.sumOf { it.count }
        val denominator = (bucket.totals[src] ?: 0.0) + smoothing * max(vocabSize, 1).toDouble()
        val best = edges.firstOrNull() ?: return null

        return NextActionPrediction(
            token = best.token,
            smoothedProbability = (best.count + smoothing) / denominator,
            observedShare = if (totalObserved > 0) best.count / totalObserved else 0.0
        )
    }

    private fun parseBucket(obj: JSONObject): Bucket {
        val totalsObj = obj.optJSONObject("totals") ?: JSONObject()
        val totals = mutableMapOf<String, Double>()
        for (k in totalsObj.keys()) totals[k] = totalsObj.getDouble(k)

        val transObj = obj.optJSONObject("transitions") ?: JSONObject()
        val transitions = mutableMapOf<String, List<Edge>>()
        for (src in transObj.keys()) {
            val edgesVal = transObj.get(src)
            val edgeList = mutableListOf<Edge>()
            if (edgesVal is JSONArray) {
                for (i in 0 until edgesVal.length()) {
                    val row = edgesVal.getJSONObject(i)
                    edgeList.add(Edge(row.getString("token"), row.getDouble("count")))
                }
            } else if (edgesVal is JSONObject) {
                for (dst in edgesVal.keys()) {
                    edgeList.add(Edge(dst, edgesVal.getDouble(dst)))
                }
            }
            transitions[src] = edgeList.sortedByDescending { it.count }
        }
        return Bucket(transitions, totals)
    }
}

// --------------------------------------------------------------------------
// 5. Cluster Business Mapping
// --------------------------------------------------------------------------

class ClusterMapping(json: String) {
    private val archetypes: Map<Long, ClusterArchetype>

    init {
        val map = mutableMapOf<Long, ClusterArchetype>()
        val obj = JSONObject(json)
        for (k in obj.keys()) {
            val item = obj.getJSONObject(k)
            val cid = item.optLong("cluster_id", k.toLongOrNull() ?: -1L)
            map[cid] = ClusterArchetype(
                clusterId = cid,
                clusterName = item.optString("cluster_name", "Archetype $cid"),
                businessFamily = item.optString("business_family", "Chưa phân loại"),
                businessSubmodule = item.optString("business_submodule", ""),
                businessDetail = item.optString("business_detail", ""),
                namingConfidence = item.optString("naming_confidence", "medium"),
                namingSource = item.optString("naming_source", ""),
                needsReview = item.optBoolean("needs_review", false),
                journeyCount = item.optInt("journey_count", 0),
                journeyShare = item.optDouble("journey_share", 0.0),
                medoidPath = item.optString("medoid_path", "")
            )
        }
        archetypes = map
    }

    fun lookup(clusterId: Long?): ClusterArchetype? {
        if (clusterId == null) return null
        return archetypes[clusterId] ?: archetypes[-1L]
    }
}

// --------------------------------------------------------------------------
// 6. Complete Journey Model Scorer
// --------------------------------------------------------------------------

class MobileJourneyModel(context: Context) : AutoCloseable {
    private val assets: AssetManager = context.assets
    private val vectorizer: MobileJourneyVectorizer
    private val centroidAssigner: MobileCentroidAssigner
    private val markov: MobileMarkovBank
    private val clusterMapping: ClusterMapping
    private val thresholds: ScoreThresholds

    init {
        val configJson = assets.readText("config.json")
        MobileJourneyPreprocessor.configure(configJson)

        val vocabJson = assets.readText("vocabularies.json")
        val metaJson = assets.readText("vectorizer_metadata.json")
        val weightsBytes = assets.readBytes("projection_weights.float16.bin")
        vectorizer = MobileJourneyVectorizer.load(vocabJson, metaJson, weightsBytes)

        val manifestJson = assets.readText("manifest.json")
        val centroidsBytes = assets.readBytes("centroids_matrix.bin")
        centroidAssigner = MobileCentroidAssigner.load(manifestJson, centroidsBytes)

        val markovJson = assets.readText("markov.json")
        markov = MobileMarkovBank(markovJson)

        val mappingJson = assets.readText("cluster_mapping.json")
        clusterMapping = ClusterMapping(mappingJson)

        val threshJson = assets.readText("thresholds.json")
        val tObj = JSONObject(threshJson)
        thresholds = ScoreThresholds(
            distanceP95 = tObj.optDouble("distance_p95", 0.70734),
            markovP05 = tObj.optDouble("markov_p05", -4.49426),
            markovP01 = tObj.optDouble("markov_p01", -5.51838),
            backRateP90 = tObj.optDouble("back_rate_p90", 0.20),
            loopsP90 = tObj.optDouble("loops_p90", 0.0),
            revisitP90 = tObj.optDouble("revisit_p90", 0.2609),
            spanP95 = tObj.optDouble("span_p95", 138.0)
        )
    }

    fun analyzeJson(jsonText: String): MobileAnalysisResult {
        val events = parseClickstreamJson(jsonText)
        return analyzeEvents(events)
    }

    fun analyzeEvents(events: List<ClickstreamEvent>): MobileAnalysisResult {
        val prepared = MobileJourneyPreprocessor.prepareAll(events)
        val predictions = prepared.map { scorePrepared(it, provisional = false) }
        return MobileAnalysisResult(predictions)
    }

    fun scoreEvents(
        journeyId: String,
        events: List<ClickstreamEvent>,
        provisional: Boolean,
        boundaryReason: String = ""
    ): MobilePrediction? {
        val prepared = MobileJourneyPreprocessor.prepareSingle(journeyId, events, boundaryReason) ?: return null
        return scorePrepared(prepared, provisional)
    }

    fun scorePrepared(prepared: PreparedJourney, provisional: Boolean): MobilePrediction {
        val isCollecting = prepared.nEventsFinal < MobileJourneyPreprocessor.minJourneyLength
        val state = when {
            isCollecting -> "collecting"
            provisional -> "provisional"
            else -> "final"
        }

        if (isCollecting) {
            return MobilePrediction(
                journeyId = prepared.journeyId,
                sessionId = prepared.sessionId,
                deviceId = prepared.deviceId,
                customerId = prepared.customerId,
                platform = prepared.platform,
                state = state,
                boundaryReason = prepared.boundaryReason,
                eventsSeen = prepared.rawEvents.size,
                eventSequence = prepared.tokens,
                cluster = null,
                clusterName = null,
                businessFamily = null,
                businessSubmodule = null,
                businessDetail = null,
                namingConfidence = null,
                distanceToCentroid = null,
                distanceLimit = thresholds.distanceP95,
                markovLogprob = null,
                geometricAnomaly = false,
                generativeAnomaly = false,
                severeAnomaly = false,
                frictionFlags = "",
                nextAction = null,
                nextActionShare = null,
                medoidPath = null
            )
        }

        // Vectorize and assign nearest centroid
        val matrix = vectorizer.transform(prepared)
        val assignment = centroidAssigner.assign(matrix.projectedEmbedding)
        val distance = assignment.distance
        val geometricAnomaly = distance > thresholds.distanceP95
        val assignedCluster = if (geometricAnomaly) -1L else assignment.clusterId

        val archetype = clusterMapping.lookup(if (geometricAnomaly) -1L else assignment.clusterId)
            ?: clusterMapping.lookup(assignment.clusterId)

        // Markov probability
        val logprob = markov.score(prepared.tokens, if (geometricAnomaly) null else assignment.clusterId)
        val generativeAnomaly = logprob != null && logprob < thresholds.markovP05
        val severeAnomaly = geometricAnomaly && logprob != null && logprob < thresholds.markovP01

        // Friction flags
        val flags = buildList {
            if (prepared.backRate > thresholds.backRateP90) add("excessive_back")
            if (prepared.nLoopRemoved > thresholds.loopsP90) add("navigation_loop")
            if (prepared.revisitRatio > thresholds.revisitP90) add("screen_thrash")
            if (prepared.spanSeconds > thresholds.spanP95) add("slow_journey")
            if (geometricAnomaly) add("unknown_archetype")
            if (generativeAnomaly) add("improbable_transitions")
        }.joinToString("|")

        // Next action prediction
        val next = markov.predictNext(prepared.tokens, if (geometricAnomaly) null else assignment.clusterId)

        return MobilePrediction(
            journeyId = prepared.journeyId,
            sessionId = prepared.sessionId,
            deviceId = prepared.deviceId,
            customerId = prepared.customerId,
            platform = prepared.platform,
            state = state,
            boundaryReason = prepared.boundaryReason,
            eventsSeen = prepared.rawEvents.size,
            eventSequence = prepared.tokens,
            cluster = assignedCluster,
            clusterName = archetype?.clusterName ?: if (geometricAnomaly) "Chưa phân loại | Journey hỗn hợp/nhiễu" else "Archetype ${assignment.clusterId}",
            businessFamily = archetype?.businessFamily,
            businessSubmodule = archetype?.businessSubmodule,
            businessDetail = archetype?.businessDetail,
            namingConfidence = archetype?.namingConfidence,
            distanceToCentroid = distance,
            distanceLimit = thresholds.distanceP95,
            markovLogprob = logprob,
            geometricAnomaly = geometricAnomaly,
            generativeAnomaly = generativeAnomaly,
            severeAnomaly = severeAnomaly,
            frictionFlags = flags,
            nextAction = next?.token,
            nextActionShare = next?.observedShare,
            medoidPath = archetype?.medoidPath
        )
    }

    override fun close() {
        // Pure Kotlin / Memory execution - no native handles to release
    }
}

// --------------------------------------------------------------------------
// Helpers
// --------------------------------------------------------------------------

private fun AssetManager.readText(name: String): String =
    open(name).bufferedReader().use { it.readText() }

private fun AssetManager.readBytes(name: String): ByteArray =
    open(name).use { it.readBytes() }

private fun JSONArray.toStringList(): List<String> =
    (0 until length()).map { getString(it) }

private fun JSONArray.toFloatArray(): FloatArray =
    FloatArray(length()) { getDouble(it).toFloat() }
