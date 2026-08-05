package vn.hifpt.clickstream

data class ClickstreamEvent(
    val sourceIndex: Int,
    val eventId: String,
    val deviceId: String,
    val customerId: String?,
    val sessionId: String,
    val createdAt: String?,
    val timestamp: Long,
    val key: String,
    val segment: String,
    val name: String,
    val screenId: String,
    val durationSeconds: Int,
    val visit: String?
)

data class ReplaySession(
    val id: String,
    val events: List<ClickstreamEvent>
) {
    val firstEvent: ClickstreamEvent get() = events.first()
    val deviceId: String get() = firstEvent.deviceId
    val platform: String get() = events.map { it.segment }.filter { it.isNotBlank() }.distinct().joinToString("/")
}
