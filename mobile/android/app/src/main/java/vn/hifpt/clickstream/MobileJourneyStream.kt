package vn.hifpt.clickstream

/**
 * Thread-safe, multi-session incremental journey stream manager.
 * Buffers events per session, detects journey boundaries, and runs scoring on completed boundaries.
 */
class MobileJourneyStream(
    private val model: MobileJourneyModel,
    private val outOfOrderToleranceMillis: Long = 2_000L,
    private val maxOpenSessions: Int = 128,
    private val maxBufferedEvents: Int = 1_000
) {
    private data class State(
        val events: MutableList<ClickstreamEvent> = mutableListOf(),
        var journeyNumber: Int = 1
    )

    private val states = linkedMapOf<String, State>()

    @Synchronized
    fun reset(sessionId: String? = null) {
        if (sessionId == null) states.clear() else states.remove(sessionId)
    }

    @Synchronized
    fun append(event: ClickstreamEvent, scoreProvisional: Boolean = true): StreamUpdate {
        if (event.sessionId.isBlank()) {
            return StreamUpdate(emptyList(), null, listOf("missing_session_id"))
        }
        val eventTime = MobileJourneyPreprocessor.eventTimeMillis(event)
        if (eventTime <= 0L) {
            return StreamUpdate(emptyList(), null, listOf("invalid_client_time"))
        }

        val evicted = mutableListOf<MobilePrediction>()
        if (event.sessionId !in states && states.size >= maxOpenSessions) {
            val oldest = states.entries.first()
            finalize(oldest.key, oldest.value, "session_evicted")?.let(evicted::add)
            states.remove(oldest.key)
        }

        val state = states.getOrPut(event.sessionId) { State() }
        val lastTime = state.events.lastOrNull()?.let(MobileJourneyPreprocessor::eventTimeMillis)
        if (lastTime != null && eventTime < lastTime - outOfOrderToleranceMillis) {
            return StreamUpdate(evicted, null, listOf("event_too_late"))
        }

        val boundary = if (lastTime == null || eventTime < lastTime) null
        else MobileJourneyPreprocessor.boundaryBefore(state.events, event)

        if (boundary != null) {
            finalize(event.sessionId, state, boundary)?.let(evicted::add)
        }

        state.events += event
        state.events.sortWith(
            compareBy<ClickstreamEvent> { MobileJourneyPreprocessor.eventTimeMillis(it) }
                .thenBy { it.sourceIndex }
        )

        if (state.events.size > maxBufferedEvents) {
            finalize(event.sessionId, state, "buffer_cap")?.let(evicted::add)
        }

        val provisional = if (scoreProvisional && state.events.isNotEmpty()) {
            model.scoreEvents(journeyId(event.sessionId, state), state.events.toList(), provisional = true)
        } else null

        return StreamUpdate(evicted, provisional)
    }

    @Synchronized
    fun snapshotProvisional(sessionId: String? = null): MobilePrediction? {
        val entry = if (sessionId == null) {
            states.entries.firstOrNull()
        } else {
            states[sessionId]?.let { java.util.AbstractMap.SimpleEntry(sessionId, it) }
        }
        return entry?.let {
            model.scoreEvents(journeyId(it.key, it.value), it.value.events.toList(), provisional = true)
        }
    }

    @Synchronized
    fun snapshotProvisionals(): List<MobilePrediction> =
        states.entries.mapNotNull { (sessionId, state) ->
            model.scoreEvents(journeyId(sessionId, state), state.events.toList(), provisional = true)
        }

    @Synchronized
    fun flush(sessionId: String, reason: String = "flush"): MobilePrediction? {
        val state = states[sessionId] ?: return null
        val prediction = finalize(sessionId, state, reason)
        states.remove(sessionId)
        return prediction
    }

    @Synchronized
    fun flush(): MobilePrediction? =
        states.keys.firstOrNull()?.let { flush(it) }

    @Synchronized
    fun flushAll(reason: String = "flush"): List<MobilePrediction> =
        states.keys.toList().mapNotNull { flush(it, reason) }

    @Synchronized
    fun flushExpired(nowEpochMs: Long): List<MobilePrediction> {
        val expired = states.filterValues { state ->
            val last = state.events.lastOrNull()?.let(MobileJourneyPreprocessor::eventTimeMillis) ?: nowEpochMs
            nowEpochMs - last > (MobileJourneyPreprocessor.idleGapSeconds * 1000.0).toLong()
        }.keys.toList()
        return expired.mapNotNull { flush(it, "idle_timeout") }
    }

    private fun finalize(sessionId: String, state: State, reason: String): MobilePrediction? {
        if (state.events.isEmpty()) return null
        val result = model.scoreEvents(
            journeyId(sessionId, state),
            state.events.toList(),
            provisional = false,
            boundaryReason = reason
        )
        state.events.clear()
        state.journeyNumber += 1
        return result
    }

    private fun journeyId(sessionId: String, state: State): String =
        "${sessionId.takeLast(8)}-J${state.journeyNumber.toString().padStart(6, '0')}"
}
