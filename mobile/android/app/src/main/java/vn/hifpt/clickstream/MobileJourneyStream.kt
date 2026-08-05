package vn.hifpt.clickstream

data class StreamUpdate(
    val finalized: List<MobilePrediction>,
    val provisional: MobilePrediction?
)

/** Incremental wrapper for simulator replay and real event ingestion. */
class MobileJourneyStream(private val model: MobileJourneyModel) {
    private val currentEvents = mutableListOf<ClickstreamEvent>()
    private var journeyNumber = 1

    fun reset() {
        currentEvents.clear()
        journeyNumber = 1
    }

    fun append(event: ClickstreamEvent, scoreProvisional: Boolean = true): StreamUpdate {
        val finalized = mutableListOf<MobilePrediction>()
        val boundary = MobileJourneyPreprocessor.boundaryBefore(currentEvents, event)
        if (boundary != null) {
            model.scoreEvents(
                journeyId = journeyId(),
                events = currentEvents.toList(),
                provisional = false,
                boundaryReason = boundary
            )?.let(finalized::add)
            currentEvents.clear()
            journeyNumber += 1
        }
        currentEvents += event
        val provisional = if (scoreProvisional) {
            model.scoreEvents(
                journeyId = journeyId(),
                events = currentEvents.toList(),
                provisional = true
            )
        } else {
            null
        }
        return StreamUpdate(finalized = finalized, provisional = provisional)
    }

    fun snapshotProvisional(): MobilePrediction? = model.scoreEvents(
        journeyId = journeyId(),
        events = currentEvents.toList(),
        provisional = true
    )

    fun flush(): MobilePrediction? {
        if (currentEvents.isEmpty()) return null
        val result = model.scoreEvents(
            journeyId = journeyId(),
            events = currentEvents.toList(),
            provisional = false,
            boundaryReason = "flush"
        )
        currentEvents.clear()
        journeyNumber += 1
        return result
    }

    private fun journeyId() = "J${journeyNumber.toString().padStart(6, '0')}"
}
