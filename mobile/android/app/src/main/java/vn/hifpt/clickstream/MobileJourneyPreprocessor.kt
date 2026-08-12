package vn.hifpt.clickstream

import org.json.JSONObject
import java.net.URI
import java.net.URLDecoder
import java.nio.charset.StandardCharsets
import java.time.Instant
import java.util.Locale
import kotlin.math.max

data class PreparedJourney(
    val journeyId: String,
    val sessionId: String,
    val deviceId: String,
    val customerId: String?,
    val platform: String,
    val boundaryReason: String,
    val rawEvents: List<ClickstreamEvent>,
    val cleanedEvents: List<CanonicalClick>,
    val tokens: List<String>,
    val numeric: Map<String, Double>,
    val nEventsFinal: Int,
    val nLoopRemoved: Int,
    val nDedupRemoved: Int,
    val backRate: Double,
    val revisitRatio: Double,
    val spanSeconds: Double
)

data class CanonicalClick(
    val raw: ClickstreamEvent,
    val eventType: String,
    val segmentName: String,
    val screenContext: String,
    val eventToken: String,
    val eventTimeMillis: Long,
    val gapPrevSeconds: Double?,
    val gapNextSeconds: Double?
)

/** Production preprocessing contract mirrored from Rule_based.production. */
object MobileJourneyPreprocessor {
    private const val MISSING = "<missing>"
    private val semanticQueryKeys = setOf("tab", "cat_id", "categoryid", "ordertype", "type", "step", "mode", "view", "status")
    private val backMarkers = listOf("btn_back", "backbutton", "handleback", "nav_back", "/back", "goback", "close", "dismiss", "cancel")
    private val authMarkers = listOf("continue_login", "login_with", "click_login", "log_out", "logout", "sign_out", "signin_success")
    private val rootNames = setOf("HomeVC", "HomeGuestVC", "HOME", "android/Home", "guest/HOME")
    private val uuidRegex = Regex("^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$", RegexOption.IGNORE_CASE)
    private val hexRegex = Regex("^[0-9a-f]{16,}$", RegexOption.IGNORE_CASE)
    private val digitRegex = Regex("^\\d+$")
    private val codeRegex = Regex("^(?=.*\\d)[A-Z0-9]{6,}$")

    @Volatile var idleGapSeconds = 90.0
        private set
    @Volatile private var rootReturnMinEvents = 6
    @Volatile private var maxJourneyLength = 80

    fun configure(preprocessingJson: String) {
        val root = JSONObject(preprocessingJson)
        val segment = root.optJSONObject("journey_contract")?.optJSONObject("segment") ?: root
        idleGapSeconds = segment.optDouble("idle_gap_seconds", idleGapSeconds)
        rootReturnMinEvents = segment.optInt("root_return_min_events", rootReturnMinEvents)
        maxJourneyLength = segment.optInt("max_journey_length", maxJourneyLength)
    }

    fun eventTimeMillis(event: ClickstreamEvent): Long =
        event.timestamp.takeIf { it > 0L }
            ?: event.createdAt?.takeIf { it.isNotBlank() }?.let { value ->
                value.toLongOrNull() ?: runCatching { Instant.parse(value).toEpochMilli() }.getOrNull()
            }
            ?: 0L

    fun canonicalize(events: List<ClickstreamEvent>): List<CanonicalClick> {
        val ordered = events.sortedWith(
            compareBy<ClickstreamEvent> { it.segment.lowercase(Locale.US) }
                .thenBy { it.sessionId }
                .thenBy { eventTimeMillis(it) }
                .thenBy { it.sourceIndex }
        )
        val result = mutableListOf<CanonicalClick>()
        var previous: CanonicalClick? = null
        var previousStream: String? = null
        var screenContext = MISSING
        var sinceCut = 0

        for (raw in ordered) {
            val time = eventTimeMillis(raw)
            val stream = streamKey(raw)
            val sameStream = previous != null && previousStream == stream
            val eventType = normalizeEventType(raw.key)
            val segmentName = canonizeName(raw.name)
            val next = ordered.getOrNull(result.size + 1)
            val sameNext = next != null && streamKey(next) == stream
            val gapPrev = if (sameStream) (time - previous!!.eventTimeMillis) / 1000.0 else null
            val provisional = makeCanonicalClick(
                raw, eventType, segmentName, screenContext, time, gapPrev,
                if (sameNext) (eventTimeMillis(next!!) - time) / 1000.0 else null
            )

            if (!sameStream) {
                screenContext = MISSING
                sinceCut = 0
            } else if (boundaryBeforeCanonical(previous!!, provisional, sinceCut) != null) {
                screenContext = MISSING
                sinceCut = 0
            }

            val canonical = makeCanonicalClick(
                raw, eventType, segmentName, screenContext, time, gapPrev,
                if (sameNext) (eventTimeMillis(next!!) - time) / 1000.0 else null
            )
            result += canonical
            if (eventType == "view") screenContext = canonical.screenContext
            previous = canonical
            previousStream = stream
            sinceCut++
        }
        return result
    }

    private fun streamKey(event: ClickstreamEvent): String =
        "${event.segment.trim().lowercase(Locale.US)}\u001f${event.sessionId}"

    private fun makeCanonicalClick(
        raw: ClickstreamEvent,
        eventType: String,
        segmentName: String,
        previousScreenContext: String,
        time: Long,
        gapPrevSeconds: Double?,
        gapNextSeconds: Double?
    ): CanonicalClick {
        val explicitScreen = raw.screenId.trim()
            .takeIf { it.isNotEmpty() }
            ?.let { canonizeName(it) }
        val screenContext = explicitScreen
            ?: if (eventType == "view") segmentName else previousScreenContext
        val eventToken = when (eventType) {
            "view" -> "$eventType@$screenContext"
            else -> "$eventType@$screenContext#$segmentName"
        }
        return CanonicalClick(
            raw = raw,
            eventType = eventType,
            segmentName = segmentName,
            screenContext = screenContext,
            eventToken = eventToken,
            eventTimeMillis = time,
            gapPrevSeconds = gapPrevSeconds,
            gapNextSeconds = gapNextSeconds
        )
    }

    fun prepareAll(events: List<ClickstreamEvent>): List<PreparedJourney> {
        val canonical = canonicalize(events)
        if (canonical.isEmpty()) return emptyList()
        val output = mutableListOf<PreparedJourney>()
        var start = 0
        var startReason = "session_start"
        for (index in 1 until canonical.size) {
            val boundary = boundaryBeforeCanonical(canonical[index - 1], canonical[index], index - start)
            if (boundary != null) {
                output += buildJourney("J${(output.size + 1).toString().padStart(6, '0')}", canonical.subList(start, index), startReason)
                start = index
                startReason = boundary
            }
        }
        output += buildJourney("J${(output.size + 1).toString().padStart(6, '0')}", canonical.subList(start, canonical.size), startReason)
        return output
    }

    fun prepareSingle(journeyId: String, events: List<ClickstreamEvent>, boundaryReason: String = ""): PreparedJourney? {
        val canonical = canonicalize(events)
        return if (canonical.isEmpty()) null else buildJourney(journeyId, canonical, boundaryReason)
    }

    fun boundaryBefore(currentEvents: List<ClickstreamEvent>, nextEvent: ClickstreamEvent): String? {
        if (currentEvents.isEmpty()) return null
        val previousRaw = currentEvents.last()
        if (previousRaw.sessionId != nextEvent.sessionId ||
            !previousRaw.segment.equals(nextEvent.segment, true)
        ) return "session_start"
        val pair = canonicalize(listOf(currentEvents.last(), nextEvent))
        if (pair.size != 2 || pair.last().raw !== nextEvent) return null
        return boundaryBeforeCanonical(pair[0], pair[1], currentEvents.size)
    }

    private fun boundaryBeforeCanonical(previous: CanonicalClick, current: CanonicalClick, sinceCut: Int): String? = when {
        current.raw.sessionId != previous.raw.sessionId ||
            !current.raw.segment.equals(previous.raw.segment, true) -> "session_start"
        (current.gapPrevSeconds ?: 0.0) > idleGapSeconds -> "idle_gap"
        sinceCut >= maxJourneyLength -> "length_cap"
        current.eventType == "view" && current.segmentName in rootNames && previous.segmentName !in rootNames && sinceCut >= rootReturnMinEvents -> "root_return"
        isAuthAction(current) && !isAuthAction(previous) -> "auth_change"
        else -> null
    }

    private fun buildJourney(journeyId: String, canonical: List<CanonicalClick>, boundaryReason: String): PreparedJourney {
        val runKept = keepAfterRuns(canonical)
        val cleaned = keepAfterCycles(runKept)
        val tokens = cleaned.map { it.eventToken }
        val n = tokens.size
        val gaps = cleaned.drop(1).mapNotNull { it.gapPrevSeconds }.filter { it >= 0.0 }
        val span = if (canonical.size > 1) max(0.0, (canonical.last().eventTimeMillis - canonical.first().eventTimeMillis) / 1000.0) else 0.0
        val unique = tokens.toSet().size
        val backRate = if (n == 0) 0.0 else cleaned.count { click -> backMarkers.any { click.segmentName.lowercase(Locale.US).contains(it) } }.toDouble() / n
        val revisit = if (n == 0) 0.0 else 1.0 - unique.toDouble() / n
        return PreparedJourney(
            journeyId, canonical.first().raw.sessionId, canonical.first().raw.deviceId,
            canonical.first().raw.customerId, canonical.first().raw.segment.lowercase(Locale.US),
            boundaryReason, canonical.map { it.raw }, cleaned, tokens,
            mapOf(
                "n_events_final" to n.toDouble(),
                "n_unique_tokens" to unique.toDouble(),
                "action_ratio" to if (n == 0) 0.0 else cleaned.count { it.eventType == "action" }.toDouble() / n,
                "back_rate" to backRate,
                "revisit_ratio" to revisit,
                "n_loop_removed" to (runKept.size - cleaned.size).toDouble(),
                "n_dedup_removed" to (canonical.size - runKept.size).toDouble(),
                "span_seconds" to span,
                "median_gap_s" to quantile(gaps, 0.5),
                "p90_gap_s" to quantile(gaps, 0.9),
                "max_gap_s" to (gaps.maxOrNull() ?: 0.0)
            ), n, runKept.size - cleaned.size, canonical.size - runKept.size, backRate, revisit, span
        )
    }

    private fun keepAfterRuns(events: List<CanonicalClick>) = events.filterIndexed { i, e -> i == 0 || e.eventToken != events[i - 1].eventToken }

    private fun keepAfterCycles(events: List<CanonicalClick>): List<CanonicalClick> {
        val kept = mutableListOf<CanonicalClick>()
        var i = 0
        while (i < events.size) {
            var matched = false
            for (period in 2..4) {
                if (i + period * 2 > events.size) continue
                val block = events.subList(i, i + period).map { it.eventToken }
                var repeats = 1
                var cursor = i + period
                while (cursor + period <= events.size && events.subList(cursor, cursor + period).map { it.eventToken } == block) {
                    repeats++; cursor += period
                }
                if (repeats >= 2) { kept += events.subList(i, i + period); i = cursor; matched = true; break }
            }
            if (!matched) kept += events[i++]
        }
        return kept
    }

    private fun normalizeEventType(value: String): String = when (value.trim().lowercase(Locale.US)) {
        "[cly]_view" -> "view"
        "[cly]_action" -> "action"
        "" -> "unknown"
        else -> value.trim().lowercase(Locale.US)
    }

    private fun isAuthAction(event: CanonicalClick) = event.eventType == "action" && authMarkers.any { event.segmentName.lowercase(Locale.US).contains(it) }

    private fun canonizeName(value: String): String {
        val text = value.trim()
        if (text.isEmpty() || text.lowercase(Locale.US) in setOf("nan", "none", "null")) return MISSING
        if (text.contains("://")) return canonizeUrl(text)
        return if ('/' in text) text.split('/').joinToString("/") { maskSegment(it) } else text
    }

    private fun canonizeUrl(value: String): String {
        val uri = runCatching { URI(value) }.getOrNull() ?: return value
        val host = uri.host ?: ""
        val path = (uri.path ?: "/").split('/').joinToString("/") { maskSegment(it) }.trimEnd('/').ifEmpty { "/" }
        val query = uri.rawQuery.orEmpty().split('&').mapNotNull { part ->
            val bits = part.split('=', limit = 2); if (bits.size != 2) return@mapNotNull null
            val key = URLDecoder.decode(bits[0], StandardCharsets.UTF_8.name()).lowercase(Locale.US)
            val decoded = URLDecoder.decode(bits[1], StandardCharsets.UTF_8.name())
            if (key in semanticQueryKeys && decoded.isNotBlank()) key to decoded else null
        }.sortedBy { it.first }.joinToString("&") { "${it.first}=${it.second}" }
        return host + path + if (query.isEmpty()) "" else "?$query"
    }

    private fun maskSegment(value: String) = when {
        digitRegex.matches(value) -> "{id}"
        uuidRegex.matches(value) || hexRegex.matches(value) -> "{uuid}"
        codeRegex.matches(value) -> "{code}"
        else -> value
    }

    private fun quantile(values: List<Double>, q: Double): Double {
        if (values.isEmpty()) return 0.0
        val sorted = values.sorted(); val position = (sorted.size - 1) * q
        val low = position.toInt(); val high = kotlin.math.ceil(position).toInt()
        return if (low == high) sorted[low] else sorted[low] + (sorted[high] - sorted[low]) * (position - low)
    }
}
