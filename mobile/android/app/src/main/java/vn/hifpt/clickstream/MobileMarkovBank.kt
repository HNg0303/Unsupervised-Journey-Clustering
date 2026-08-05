package vn.hifpt.clickstream

import org.json.JSONObject
import kotlin.math.ln

data class NextActionPrediction(
    val token: String,
    val smoothedProbability: Double,
    val observedShare: Double
)

/** Sparse first-order Markov companion exported from the fitted Python scorer. */
class MobileMarkovBank(markovJson: String) {
    private data class Edge(val token: String, val count: Double)
    private data class Bucket(
        val transitions: Map<String, List<Edge>>,
        val totals: Map<String, Double>
    )

    private val smoothing: Double
    private val vocabulary: Set<String>
    private val global: Bucket
    private val clusters: Map<Long, Bucket>

    init {
        val root = JSONObject(markovJson)
        smoothing = root.optDouble("smoothing", 0.5)
        vocabulary = root.getJSONArray("vocabulary").toStringList().toSet()
        global = parseBucket(root.getJSONObject("global"))
        val clusterObject = root.optJSONObject("clusters") ?: JSONObject()
        clusters = clusterObject.keys().asSequence().associate { key ->
            key.toLong() to parseBucket(clusterObject.getJSONObject(key))
        }
    }

    fun score(tokens: List<String>, cluster: Long): Double? {
        if (tokens.size < 2) return null
        val bucket = clusters[cluster] ?: global
        val denominatorMass = smoothing * vocabulary.size.toDouble()
        var totalLogProbability = 0.0
        var pairs = 0
        for (index in 0 until tokens.lastIndex) {
            val source = tokens[index]
            val target = tokens[index + 1]
            if (source !in vocabulary || target !in vocabulary) continue
            val count = bucket.transitions[source]
                ?.firstOrNull { it.token == target }
                ?.count ?: 0.0
            val denominator = (bucket.totals[source] ?: 0.0) + denominatorMass
            totalLogProbability += ln(count + smoothing) - ln(denominator)
            pairs += 1
        }
        return if (pairs == 0) null else totalLogProbability / pairs
    }

    fun predictNext(tokens: List<String>, cluster: Long, topK: Int = 3): NextActionPrediction? {
        val source = tokens.lastOrNull() ?: return null
        if (source !in vocabulary) return null
        val bucket = clusters[cluster] ?: global
        val edges = bucket.transitions[source].orEmpty()
        if (edges.isEmpty()) return null
        val totalObserved = edges.sumOf { it.count }
        val denominator = (bucket.totals[source] ?: 0.0) + smoothing * vocabulary.size.toDouble()
        return edges.take(topK).firstOrNull()?.let { edge ->
            NextActionPrediction(
                token = edge.token,
                smoothedProbability = (edge.count + smoothing) / denominator,
                observedShare = if (totalObserved == 0.0) 0.0 else edge.count / totalObserved
            )
        }
    }

    private fun parseBucket(value: JSONObject): Bucket {
        val totalsObject = value.optJSONObject("totals") ?: JSONObject()
        val totals = totalsObject.keys().asSequence().associateWith { totalsObject.getDouble(it) }
        val transitionObject = value.optJSONObject("transitions") ?: JSONObject()
        val transitions = transitionObject.keys().asSequence().associateWith { source ->
            val rows = transitionObject.getJSONArray(source)
            (0 until rows.length()).map { index ->
                val row = rows.getJSONObject(index)
                Edge(row.getString("token"), row.getDouble("count"))
            }
        }
        return Bucket(transitions, totals)
    }
}

private fun org.json.JSONArray.toStringList() = (0 until length()).map { getString(it) }
