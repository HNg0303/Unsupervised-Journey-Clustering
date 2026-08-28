package vn.hifpt.clickstream

import android.content.Context
import org.json.JSONArray
import org.json.JSONObject
import java.time.Instant
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

    /** Parse the production CSV schema used by android_t3_1k.csv and test_data_android.csv. */
    fun parseCsvEvents(csv: String): List<ClickstreamEvent> {
        val rows = parseCsvRows(csv)
        if (rows.isEmpty()) return emptyList()
        val header = rows.first().map { it.trim().removePrefix("\uFEFF") }
        val index = header.withIndex().associate { it.value to it.index }

        fun value(row: List<String>, vararg names: String): String {
            for (name in names) {
                val idx = index[name]
                if (idx != null) {
                    val v = row.getOrNull(idx).orEmpty().trim()
                    if (v.isNotEmpty()) return v
                }
            }
            return ""
        }

        return rows.drop(1).mapIndexedNotNull { sequence, row ->
            if (row.all { it.isBlank() }) return@mapIndexedNotNull null
            val clientTime = value(row, "client_time", "timestamp")
            val timestamp = parseClientTimeToMillis(clientTime)
            val sessionId = value(row, "session_id")
            if (sessionId.isBlank()) return@mapIndexedNotNull null

            val customerIdRaw = value(row, "customer_id")
            val customerId = if (customerIdRaw.isNotBlank() && customerIdRaw.lowercase(Locale.US) !in setOf("null", "none", "nan")) {
                if (customerIdRaw.endsWith(".0") && customerIdRaw.dropLast(2).all { it.isDigit() }) {
                    customerIdRaw.dropLast(2)
                } else customerIdRaw
            } else null

            ClickstreamEvent(
                sourceIndex = sequence,
                eventId = value(row, "_id", "event_id").ifBlank { "row-$sequence" },
                deviceId = value(row, "device_id"),
                customerId = customerId,
                sessionId = sessionId,
                createdAt = value(row, "device_created_at", "created_at").takeIf { it.isNotBlank() } ?: clientTime,
                timestamp = timestamp,
                key = value(row, "key").ifBlank { "unknown" },
                segment = value(row, "platform", "segmentation_segment", "segment").ifBlank { "Android" },
                name = value(row, "segmentation_name", "name").ifBlank { value(row, "key", "unknown") },
                screenId = value(row, "screen_id", "segmentation_screen_id"),
                durationSeconds = value(row, "dur", "duration").toDoubleOrNull()?.toInt() ?: 0,
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
    val trimmed = json.trim()
    if (trimmed.isEmpty()) return emptyList()
    val rows = if (trimmed.startsWith("[")) JSONArray(trimmed) else JSONArray().put(JSONObject(trimmed))
    return ArrayList<ClickstreamEvent>(rows.length()).apply {
        for (index in 0 until rows.length()) {
            add(rows.getJSONObject(index).toEvent(index))
        }
    }
}

private fun JSONObject.toEvent(sequence: Int): ClickstreamEvent {
    val segmentation = optJSONObject("segmentation") ?: JSONObject()
    val eventId = optJSONObject("_id")?.optString("\$oid")
        ?.takeIf { it.isNotBlank() }
        ?: optString("_id").takeIf { it.isNotBlank() }
        ?: "row-$sequence"

    val clientTimeValue = opt("client_time") ?: opt("timestamp")
    val clientTimeText = when (clientTimeValue) {
        is String -> clientTimeValue
        is Number -> clientTimeValue.toLong().toString()
        else -> null
    }
    val createdAt = clientTimeText ?: optJSONObject("created_at")?.optString("\$date")
    val eventTimestamp = parseClientTimeToMillis(clientTimeValue)

    val customerId = if (has("customer_id") && !isNull("customer_id")) {
        val raw = opt("customer_id")?.toString()?.trim()
        if (raw.isNullOrBlank() || raw.lowercase(Locale.US) in setOf("null", "none", "nan")) null
        else if (raw.endsWith(".0") && raw.dropLast(2).all { it.isDigit() }) raw.dropLast(2)
        else raw
    } else null

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
            optString("segmentation.segment"),
            "Android"
        ),
        name = firstNonBlank(
            optString("segmentation_name"),
            segmentation.optString("name"),
            optString("segmentation.name"),
            optString("key", "unknown")
        ),
        screenId = firstNonBlank(
            optString("screen_id"),
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

private fun parseClientTimeToMillis(value: Any?): Long = when (value) {
    null -> 0L
    is Number -> value.toLong()
    is String -> {
        val s = value.trim()
        s.toLongOrNull() ?: runCatching { Instant.parse(s).toEpochMilli() }.getOrDefault(0L)
    }
    else -> 0L
}

private fun firstNonBlank(vararg values: String): String =
    values.firstOrNull { it.isNotBlank() } ?: ""

private fun firstNonBlankOrNull(vararg values: String?): String? =
    values.firstOrNull { !it.isNullOrBlank() }
