package vn.hifpt.clickstream

import android.content.Context
import org.json.JSONArray
import org.json.JSONObject

class ClickstreamRepository(private val context: Context) {
    fun loadSessions(assetName: String = "test_data_july.json"): List<ReplaySession> {
        val json = context.assets.open(assetName).bufferedReader().use { it.readText() }
        val events = parseEvents(json)

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

    /** Parse the same event array from an asset, file picker, or replay payload. */
    fun parseEvents(json: String): List<ClickstreamEvent> {
        return parseClickstreamJson(json)
    }
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
        ?: "row-$sequence"
    val createdAt = optJSONObject("created_at")?.optString("\$date")
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
        timestamp = optLong("timestamp"),
        key = optString("key", "unknown"),
        segment = firstNonBlank(
            segmentation.optString("segment"),
            optString("segmentation.segment")
        ),
        name = firstNonBlank(
            segmentation.optString("name"),
            optString("segmentation.name"),
            optString("key", "unknown")
        ),
        screenId = firstNonBlank(
            segmentation.optString("screen_id"),
            optString("segmentation.screen_id")
        ),
        durationSeconds = firstInt(segmentation.opt("dur"), opt("dur")),
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
