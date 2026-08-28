package vn.hifpt.clickstream

import org.json.JSONArray
import org.json.JSONObject

/**
 * Production clickstream processor facade supporting FIXED_DURATION and JOURNEY_COMPLETE routes.
 * Receives live clickstream events or raw JSON payloads and dispatches updates.
 */
class MobileClickstreamProcessor(
    private val model: MobileJourneyModel,
    val route: MobileProcessingRoute = MobileProcessingRoute.JOURNEY_COMPLETE,
    val windowDurationSeconds: Long = 30L
) {
    private val stream = MobileJourneyStream(model)
    private var windowStartIndex: Int = 1
    private var windowStartTimestamp: Long? = null
    private var windowEventsCount: Int = 0
    private val finalizedInWindow = mutableListOf<MobilePrediction>()

    fun start() {
        stream.reset()
        windowStartIndex = 1
        windowStartTimestamp = null
        windowEventsCount = 0
        finalizedInWindow.clear()
    }

    @Synchronized
    fun accept(event: ClickstreamEvent): MobileProcessingUpdate {
        val eventTime = MobileJourneyPreprocessor.eventTimeMillis(event)
        if (windowStartTimestamp == null && eventTime > 0L) {
            windowStartTimestamp = eventTime
        }

        val update = stream.append(event, scoreProvisional = (route == MobileProcessingRoute.FIXED_DURATION))
        finalizedInWindow.addAll(update.finalized)
        windowEventsCount++

        if (route == MobileProcessingRoute.JOURNEY_COMPLETE) {
            return MobileProcessingUpdate(
                route = route,
                windowIndex = null,
                eventsInWindow = 1,
                windowStartTimestamp = eventTime,
                windowEndTimestamp = eventTime,
                finalized = update.finalized,
                provisional = emptyList(),
                errors = update.errors
            )
        }

        // FIXED_DURATION route
        val start = windowStartTimestamp ?: eventTime
        if (eventTime - start >= windowDurationSeconds * 1000L) {
            val emittedFinalized = finalizedInWindow.toList()
            finalizedInWindow.clear()
            val emittedProvisional = stream.snapshotProvisionals()
            val wIndex = windowStartIndex++
            val wStart = start
            windowStartTimestamp = eventTime
            val count = windowEventsCount
            windowEventsCount = 0

            return MobileProcessingUpdate(
                route = route,
                windowIndex = wIndex,
                eventsInWindow = count,
                windowStartTimestamp = wStart,
                windowEndTimestamp = eventTime,
                finalized = emittedFinalized,
                provisional = emittedProvisional,
                errors = update.errors
            )
        }

        return MobileProcessingUpdate(
            route = route,
            windowIndex = null,
            eventsInWindow = windowEventsCount,
            windowStartTimestamp = start,
            windowEndTimestamp = eventTime,
            finalized = emptyList(),
            provisional = emptyList(),
            errors = update.errors
        )
    }

    @Synchronized
    fun acceptJson(rawJson: String): List<MobileProcessingUpdate> {
        val trimmed = rawJson.trim()
        if (trimmed.isEmpty()) return emptyList()
        val events = if (trimmed.startsWith("[")) {
            parseClickstreamJson(trimmed)
        } else {
            val arr = JSONArray().put(JSONObject(trimmed))
            parseClickstreamJson(arr.toString())
        }
        return events.map { accept(it) }.filter { it.emitted }
    }

    @Synchronized
    fun finish(sessionId: String? = null): MobileProcessingUpdate? {
        val flushed = if (sessionId != null) {
            listOfNotNull(stream.flush(sessionId, "session_end"))
        } else {
            stream.flushAll("stream_finished")
        }

        finalizedInWindow.addAll(flushed)
        if (finalizedInWindow.isEmpty() && windowEventsCount == 0) return null

        val emittedFinalized = finalizedInWindow.toList()
        finalizedInWindow.clear()

        return MobileProcessingUpdate(
            route = route,
            windowIndex = if (route == MobileProcessingRoute.FIXED_DURATION) windowStartIndex++ else null,
            eventsInWindow = windowEventsCount,
            windowStartTimestamp = windowStartTimestamp,
            windowEndTimestamp = System.currentTimeMillis(),
            finalized = emittedFinalized,
            provisional = emptyList(),
            errors = emptyList()
        )
    }
}
