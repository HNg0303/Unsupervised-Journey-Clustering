package vn.hifpt.journey

import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import org.json.JSONObject
import java.nio.FloatBuffer
import kotlin.math.ln
import kotlin.math.sqrt

data class JourneyFeatures(
    val tokens: List<String>,
    val numeric: Map<String, Double>
)

data class JourneyPrediction(
    val cluster: Long,
    val classGroup: String,
    val classCode: String,
    val className: String,
    val distance: Float,
    val geometricAnomaly: Boolean,
    val provisional: Boolean
)

/** Mobile inference for the bundle produced by export_scorer_to_onnx.py. */
class JourneyOnnxClassifier(
    modelBytes: ByteArray,
    preprocessingJson: String,
    classMappingJson: String
) : AutoCloseable {
    private val environment = OrtEnvironment.getEnvironment()
    private val session: OrtSession = environment.createSession(modelBytes)
    private val prep = JSONObject(preprocessingJson)
    private val vocabulary: List<String>
    private val vocabIndex: Map<String, Int>
    private val idf: DoubleArray
    private val svd: Array<DoubleArray>
    private val numericColumns: List<String>
    private val numericMean: DoubleArray
    private val numericScale: DoubleArray
    private val logColumns: Set<String>
    private val numericWeight: Double
    private val classes: Map<Long, JSONObject>

    init {
        vocabulary = prep.getJSONArray("vocabulary").toStringList()
        vocabIndex = vocabulary.withIndex().associate { it.value to it.index }
        idf = prep.getJSONArray("idf").toDoubleArray()
        svd = prep.getJSONArray("svd_components").toDoubleMatrix()
        numericColumns = prep.getJSONArray("numeric_columns").toStringList()
        numericMean = prep.getJSONArray("numeric_mean").toDoubleArray()
        numericScale = prep.getJSONArray("numeric_scale").toDoubleArray()
        logColumns = prep.getJSONArray("numeric_log1p_columns").toStringList().toSet()
        numericWeight = prep.getDouble("numeric_block_weight")
        val rows = JSONObject(classMappingJson).getJSONArray("classes")
        classes = (0 until rows.length()).associate { index ->
            val row = rows.getJSONObject(index)
            row.getLong("cluster") to row
        }
    }

    fun predict(input: JourneyFeatures, provisional: Boolean = false): JourneyPrediction {
        val feature = buildFeature(input)
        val tensor = OnnxTensor.createTensor(
            environment, FloatBuffer.wrap(feature), longArrayOf(1, feature.size.toLong())
        )
        tensor.use {
            session.run(mapOf("features" to tensor)).use { result ->
                val cluster = (result.get("cluster").get().value as LongArray)[0]
                val distance = (result.get("distance").get().value as FloatArray)[0]
                val anomaly = (result.get("geometric_anomaly").get().value as BooleanArray)[0]
                val label = classes[cluster] ?: classes.getValue(-1L)
                return JourneyPrediction(
                    cluster, label.getString("class_group"), label.getString("class_code"),
                    label.getString("class_name"), distance, anomaly, provisional
                )
            }
        }
    }

    private fun buildFeature(input: JourneyFeatures): FloatArray {
        val bounded = listOf("<bos>") + input.tokens + "<eos>"
        val counts = DoubleArray(vocabulary.size)
        for (n in 1..3) for (start in 0..bounded.size - n) {
            val term = bounded.subList(start, start + n).joinToString(" ")
            vocabIndex[term]?.let { counts[it] += 1.0 }
        }
        for (i in counts.indices) if (counts[i] > 0.0) {
            counts[i] = (1.0 + ln(counts[i])) * idf[i]
        }
        normalize(counts)
        val sequence = DoubleArray(svd.size) { component -> dot(svd[component], counts) }
        normalize(sequence)

        val numeric = DoubleArray(numericColumns.size) { i ->
            val column = numericColumns[i]
            var value = input.numeric[column] ?: 0.0
            if (column in logColumns) value = kotlin.math.ln1p(value.coerceAtLeast(0.0))
            (value - numericMean[i]) / numericScale[i]
        }
        normalize(numeric)
        return FloatArray(sequence.size + numeric.size) { index ->
            if (index < sequence.size) sequence[index].toFloat()
            else (numeric[index - sequence.size] * numericWeight).toFloat()
        }
    }

    override fun close() = session.close()

    private fun normalize(values: DoubleArray) {
        val norm = sqrt(values.sumOf { it * it })
        if (norm > 0.0) for (i in values.indices) values[i] /= norm
    }
    private fun dot(left: DoubleArray, right: DoubleArray) = left.indices.sumOf { left[it] * right[it] }
}

private fun org.json.JSONArray.toStringList() = (0 until length()).map { getString(it) }
private fun org.json.JSONArray.toDoubleArray() = DoubleArray(length()) { getDouble(it) }
private fun org.json.JSONArray.toDoubleMatrix() = Array(length()) { getJSONArray(it).toDoubleArray() }

