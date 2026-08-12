package vn.hifpt.clickstream

import android.content.Context
import org.json.JSONArray
import org.json.JSONObject
import java.util.Locale

class ClickstreamRepository(private val context: Context) {
    fun loadSessions(assetName: String = "test_data_android.csv"): List<ReplaySession> {
        val source = context.assets.open(assetName).bufferedReader().use { it.readText() }
        val events = if (assetName.lowercase(Locale.US).endsWith(".csv")) {
            parseCsvEvents(source)
        } else {
            parseEvents(source)
        }

        return events
            .filter { it.sessionId.isNotBlank() }
            .groupBy { it.sessionId }
            .map { (sessionId, sessionEvents) ->
                ReplaySession(
                    id = sessionId,
                    events = sessionEvents.sortedWith(
                        compareBy<ClickstreamEvent> { it.timestamp }
                            .thenBy { it.sourceIndex }
                    )
                )
            }
            .sortedByDescending { it.events.size }
    }

    /** Parse JSON event arrays from an asset, file picker, or replay payload. */
    fun parseEvents(json: String): List<ClickstreamEvent> {
        return parseClickstreamJson(json)
    }

    /** Parse the raw production CSV schema used by data/test_data/test_data_android.csv. */
    fun parseCsvEvents(csv: String): List<ClickstreamEvent> {
        val rows = parseCsvRows(csv)
        if (rows.isEmpty()) return emptyList()
        val header = rows.first().map { it.trim().removePrefix("\uFEFF") }
        val index = header.withIndex().associate { it.value to it.index }
        fun value(row: List<String>, name: String): String =
            index[name]?.let { row.getOrNull(it).orEmpty() }.orEmpty()

        return rows.drop(1).mapIndexedNotNull { sequence, row ->
            if (row.all { it.isBlank() }) return@mapIndexedNotNull null
            val clientTime = value(row, "client_time")
            ClickstreamEvent(
                sourceIndex = sequence,
                eventId = "row-$sequence",
                deviceId = value(row, "device_id"),
                customerId = value(row, "customer_id").takeIf { it.isNotBlank() },
                sessionId = value(row, "session_id"),
                createdAt = value(row, "device_created_at").takeIf { it.isNotBlank() },
                timestamp = clientTime.toLongOrNull() ?: 0L,
                key = value(row, "key").ifBlank { "unknown" },
                segment = value(row, "platform"),
                name = value(row, "segmentation_name").ifBlank { "unknown" },
                screenId = value(row, "screen_id"),
                durationSeconds = value(row, "dur").toDoubleOrNull()?.toInt() ?: 0,
                visit = value(row, "visit").takeIf { it.isNotBlank() }
            )
        }
    }
}

private fun parseCsvRows(csv: String): List<List<String>> = buildList {
    val row = mutableListOf<String>()
    val field = StringBuilder()
    var quoted = false
    var index = 0

    fun finishField() {
        row += field.toString()
        field.clear()
    }

    fun finishRow() {
        finishField()
        if (row.isNotEmpty()) add(row.toList())
        row.clear()
    }

    while (index < csv.length) {
        when (val char = csv[index]) {
            '"' -> if (quoted && index + 1 < csv.length && csv[index + 1] == '"') {
                field.append('"')
                index++
            } else {
                quoted = !quoted
            }
            ',' -> if (quoted) field.append(char) else finishField()
            '\n' -> if (quoted) field.append(char) else finishRow()
            '\r' -> Unit
            else -> field.append(char)
        }
        index++
    }
    if (field.isNotEmpty() || row.isNotEmpty()) finishRow()
}

fun parseClickstreamJson(json: String): List<ClickstreamEvent> {
    val rows = JSONArray(json)
    return ArrayList<ClickstreamEvent>(rows.length()).apply {
        for (index in 0 until rows.length()) add(rows.getJSONObject(index).toEvent(index))
    }
}

private fun JSONObject.toEvent(sequence: Int): ClickstreamEvent {
    val segmentation = optJSONObject("segmentation") ?: JSONObject()
    val eventId = optJSONObject("_id")?.optString("\$oid")
        ?.takeIf { it.isNotBlank() }
        ?: optString("_id").takeIf { it.isNotBlank() }
        ?: "row-$sequence"
    val clientTimeValue = opt("client_time")
    val clientTimeText = when (clientTimeValue) {
        is String -> clientTimeValue
        is Number -> clientTimeValue.toLong().toString()
        else -> null
    }
    val createdAt = clientTimeText ?: optJSONObject("created_at")?.optString("\$date")
    val eventTimestamp = when (clientTimeValue) {
        is Number -> clientTimeValue.toLong()
        is String -> clientTimeValue.toLongOrNull() ?: runCatching { java.time.Instant.parse(clientTimeValue).toEpochMilli() }.getOrDefault(0L)
        else -> optLong("timestamp")
    }
    val customerId = if (has("customer_id") && !isNull("customer_id")) {
        opt("customer_id")?.toString()
    } else {
        null
    }

    return ClickstreamEvent(
        sourceIndex = sequence,
        eventId = eventId,
        deviceId = optString("device_id"),
        customerId = customerId,
        sessionId = optString("session_id"),
        createdAt = createdAt,
        timestamp = eventTimestamp,
        key = optString("key", "unknown"),
        segment = firstNonBlank(
            optString("platform"),
            segmentation.optString("segment"),
            optString("segmentation.segment")
        ),
        name = firstNonBlank(
            optString("segmentation_name"),
            segmentation.optString("name"),
            optString("segmentation.name"),
            optString("key", "unknown")
        ),
        screenId = firstNonBlank(
            segmentation.optString("screen_id"),
            optString("segmentation.screen_id")
        ),
        durationSeconds = 0,
        visit = firstNonBlankOrNull(
            segmentation.opt("visit")?.toString(),
            optString("segmentation.visit")
        )
    )
}

private fun firstNonBlank(vararg values: String): String =
    values.firstOrNull { it.isNotBlank() } ?: ""

private fun firstNonBlankOrNull(vararg values: String?): String? =
    values.firstOrNull { !it.isNullOrBlank() }

private fun firstInt(vararg values: Any?): Int = values
    .firstNotNullOfOrNull { value ->
        when (value) {
            is Number -> value.toInt()
            is String -> value.toIntOrNull()
            else -> null
        }
    }
    ?: 0
