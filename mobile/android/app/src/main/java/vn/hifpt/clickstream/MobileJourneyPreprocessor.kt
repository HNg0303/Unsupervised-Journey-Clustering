package vn.hifpt.clickstream

import org.json.JSONObject
import java.net.URI
import java.net.URLDecoder
import java.nio.charset.StandardCharsets
import java.time.Instant
import java.util.Locale
import kotlin.math.max

/**
 * Production clickstream preprocessing and journey segmentation.
 * Matches Python pipeline stages: canonize -> enrich -> tokenize -> segment -> postprocess.
 */
object MobileJourneyPreprocessor {
    const val MISSING = "<missing>"
    const val TOKEN_SEPARATOR = "@"

    private val semanticQueryKeys = setOf(
        "tab", "cat_id", "categoryid", "ordertype", "type", "step", "mode", "view", "status"
    )
    private val backMarkers = listOf(
        "btn_back", "backbutton", "handleback", "nav_back", "/back", "goback", "close", "dismiss", "cancel"
    )
    private val authMarkers = listOf(
        "continue_login", "login_with", "click_login", "log_out", "logout", "sign_out", "signin_success"
    )
    private val rootNames = setOf(
        "HomeVC", "HomeGuestVC", "HOME", "android/Home", "guest/HOME"
    )

    private val uuidRegex = Regex("^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$", RegexOption.IGNORE_CASE)
    private val hexRegex = Regex("^[0-9a-f]{16,}$", RegexOption.IGNORE_CASE)
    private val digitRegex = Regex("^\\d+$")
    private val codeRegex = Regex("^(?=.*\\d)[A-Z0-9]{6,}$")

    @Volatile var idleGapSeconds = 90.0
        private set
    @Volatile var rootReturnMinEvents = 6
        private set
    @Volatile var maxJourneyLength = 80
        private set
    @Volatile var minJourneyLength = 4
        private set
    @Volatile var cutOnRootReturn = true
        private set
    @Volatile var cutOnAuthChange = true
        private set

    fun configure(configJson: String) {
        val root = JSONObject(configJson)
        val segment = root.optJSONObject("segment") ?: root.optJSONObject("journey_contract")?.optJSONObject("segment") ?: root
        idleGapSeconds = segment.optDouble("idle_gap_seconds", idleGapSeconds)
        rootReturnMinEvents = segment.optInt("root_return_min_events", rootReturnMinEvents)
        maxJourneyLength = segment.optInt("max_journey_length", maxJourneyLength)
        minJourneyLength = segment.optInt("min_journey_length", minJourneyLength)
        cutOnRootReturn = segment.optBoolean("cut_on_root_return", cutOnRootReturn)
        cutOnAuthChange = segment.optBoolean("cut_on_auth_change", cutOnAuthChange)
    }

    fun eventTimeMillis(event: ClickstreamEvent): Long {
        if (event.timestamp > 0L) return event.timestamp
        val text = event.createdAt?.trim() ?: return 0L
        if (text.isEmpty()) return 0L
        return text.toLongOrNull() ?: runCatching { Instant.parse(text).toEpochMilli() }.getOrDefault(0L)
    }

    /** Attach screen context, canonical paths, and multi-resolution semantic tokens. */
    fun canonicalize(events: List<ClickstreamEvent>): List<CanonicalClick> {
        val ordered = events.sortedWith(
            compareBy<ClickstreamEvent> { it.segment.lowercase(Locale.US) }
                .thenBy { it.sessionId }
                .thenBy { eventTimeMillis(it) }
                .thenBy { it.sourceIndex }
        )
        val result = mutableListOf<CanonicalClick>()
        var previousStream: String? = null
        var previousClick: CanonicalClick? = null
        var currentScreenContext = MISSING
        var sinceCut = 0

        for (i in ordered.indices) {
            val raw = ordered[i]
            val time = eventTimeMillis(raw)
            val stream = streamKey(raw)
            val sameStream = previousStream == stream
            val eventType = normalizeEventType(raw.key)
            val segmentName = canonizeName(raw.name)
            val nextRaw = ordered.getOrNull(i + 1)
            val sameNext = nextRaw != null && streamKey(nextRaw) == stream

            val gapPrev = if (sameStream && previousClick != null) {
                (time - previousClick!!.eventTimeMillis) / 1000.0
            } else null
            val gapNext = if (sameNext && nextRaw != null) {
                (eventTimeMillis(nextRaw) - time) / 1000.0
            } else null

            // Boundary check for screen context reset
            val boundary = if (sameStream && previousClick != null) {
                boundaryBeforeCanonical(previousClick!!, eventType, segmentName, gapPrev, sinceCut)
            } else null

            if (!sameStream || boundary != null) {
                currentScreenContext = MISSING
                sinceCut = 0
            }

            val explicitScreen = raw.screenId.trim()
                .takeIf { it.isNotEmpty() && it.lowercase(Locale.US) !in setOf("nan", "none", "null", "<missing>") }
                ?.let { canonizeName(it) }

            val screen = when {
                explicitScreen != null -> explicitScreen
                eventType == "view" -> segmentName
                else -> currentScreenContext
            }

            val eventToken = if (eventType == "view") {
                "view$TOKEN_SEPARATOR$screen"
            } else {
                "action$TOKEN_SEPARATOR$screen#$segmentName"
            }

            // Semantic enrichment
            val semantics = MobileSemanticTaxonomy.classifyEvent(eventType, screen, segmentName)
            val tokenL1 = semantics.businessFamily
            val tokenL2 = "${semantics.businessFamily}/${semantics.businessModule}"
            val tokenL3 = "${semantics.businessFamily}/${semantics.businessModule}/${semantics.businessObject}/${semantics.operation}"
            val operationToken = "${semantics.operationStage}:${semantics.operation}"

            val canonical = CanonicalClick(
                raw = raw,
                eventType = eventType,
                segmentName = segmentName,
                screenContext = screen,
                eventToken = eventToken,
                eventTimeMillis = time,
                gapPrevSeconds = gapPrev,
                gapNextSeconds = gapNext,
                semantics = semantics,
                tokenL1 = tokenL1,
                tokenL2 = tokenL2,
                tokenL3 = tokenL3,
                operationToken = operationToken
            )

            result.add(canonical)
            if (eventType == "view") {
                currentScreenContext = screen
            }
            previousStream = stream
            previousClick = canonical
            sinceCut++
        }
        return result
    }

    private fun streamKey(event: ClickstreamEvent): String =
        "${event.segment.trim().lowercase(Locale.US)}\u001f${event.sessionId}"

    fun prepareAll(events: List<ClickstreamEvent>): List<PreparedJourney> {
        val canonical = canonicalize(events)
        if (canonical.isEmpty()) return emptyList()
        val output = mutableListOf<PreparedJourney>()
        var start = 0
        var startReason = "session_start"

        for (index in 1 until canonical.size) {
            val prev = canonical[index - 1]
            val curr = canonical[index]
            val boundary = boundaryBetween(prev, curr, index - start)
            if (boundary != null) {
                output += buildJourney(
                    "J${(output.size + 1).toString().padStart(6, '0')}",
                    canonical.subList(start, index),
                    startReason
                )
                start = index
                startReason = boundary
            }
        }
        output += buildJourney(
            "J${(output.size + 1).toString().padStart(6, '0')}",
            canonical.subList(start, canonical.size),
            startReason
        )
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
            !previousRaw.segment.equals(nextEvent.segment, ignoreCase = true)
        ) return "session_start"

        val pair = canonicalize(listOf(previousRaw, nextEvent))
        if (pair.size != 2 || pair.last().raw !== nextEvent) return null
        return boundaryBetween(pair[0], pair[1], currentEvents.size)
    }

    private fun boundaryBetween(previous: CanonicalClick, current: CanonicalClick, sinceCut: Int): String? = when {
        current.raw.sessionId != previous.raw.sessionId ||
            !current.raw.segment.equals(previous.raw.segment, ignoreCase = true) -> "session_start"
        (current.gapPrevSeconds ?: 0.0) > idleGapSeconds -> "idle_gap"
        sinceCut >= maxJourneyLength -> "length_cap"
        cutOnRootReturn && current.eventType == "view" && current.segmentName in rootNames && previous.segmentName !in rootNames && sinceCut >= rootReturnMinEvents -> "root_return"
        cutOnAuthChange && isAuthAction(current.eventType, current.segmentName) && !isAuthAction(previous.eventType, previous.segmentName) -> "auth_change"
        else -> null
    }

    private fun boundaryBeforeCanonical(
        previous: CanonicalClick,
        currentEventType: String,
        currentSegmentName: String,
        gapPrevSeconds: Double?,
        sinceCut: Int
    ): String? = when {
        (gapPrevSeconds ?: 0.0) > idleGapSeconds -> "idle_gap"
        sinceCut >= maxJourneyLength -> "length_cap"
        cutOnRootReturn && currentEventType == "view" && currentSegmentName in rootNames && previous.segmentName !in rootNames && sinceCut >= rootReturnMinEvents -> "root_return"
        cutOnAuthChange && isAuthAction(currentEventType, currentSegmentName) && !isAuthAction(previous.eventType, previous.segmentName) -> "auth_change"
        else -> null
    }

    private fun isAuthAction(eventType: String, name: String): Boolean =
        eventType == "action" && authMarkers.any { name.lowercase(Locale.US).contains(it) }

    private fun buildJourney(journeyId: String, canonical: List<CanonicalClick>, boundaryReason: String): PreparedJourney {
        val runKept = keepAfterRuns(canonical)
        val cleaned = keepAfterCycles(runKept)
        val tokens = cleaned.map { it.eventToken }
        val coarseTokens = cleaned.map { it.tokenL2 }
        val intentTokens = cleaned.map { it.tokenL3 }
        val operationTokens = cleaned.map { it.operationToken }

        val n = tokens.size
        val gaps = cleaned.drop(1).mapNotNull { it.gapPrevSeconds }.filter { it >= 0.0 }
        val span = if (canonical.size > 1) {
            max(0.0, (canonical.last().eventTimeMillis - canonical.first().eventTimeMillis) / 1000.0)
        } else 0.0
        val unique = tokens.toSet().size
        val backRate = if (n == 0) 0.0 else cleaned.count { click ->
            backMarkers.any { click.segmentName.lowercase(Locale.US).contains(it) }
        }.toDouble() / n
        val revisit = if (n == 0) 0.0 else 1.0 - unique.toDouble() / n
        val actionRatio = if (n == 0) 0.0 else cleaned.count { it.eventType == "action" }.toDouble() / n

        val numericMap = mapOf(
            "n_events_final" to n.toDouble(),
            "n_unique_tokens" to unique.toDouble(),
            "action_ratio" to actionRatio,
            "back_rate" to backRate,
            "revisit_ratio" to revisit,
            "n_loop_removed" to (runKept.size - cleaned.size).toDouble(),
            "n_dedup_removed" to (canonical.size - runKept.size).toDouble(),
            "span_seconds" to span,
            "median_gap_s" to quantile(gaps, 0.5),
            "p90_gap_s" to quantile(gaps, 0.9),
            "max_gap_s" to (gaps.maxOrNull() ?: 0.0)
        )

        val channels = mapOf(
            "primary" to tokens,
            "coarse" to coarseTokens,
            "intent" to intentTokens,
            "operation" to operationTokens
        )

        return PreparedJourney(
            journeyId = journeyId,
            sessionId = canonical.first().raw.sessionId,
            deviceId = canonical.first().raw.deviceId,
            customerId = canonical.first().raw.customerId,
            platform = canonical.first().raw.segment.lowercase(Locale.US),
            boundaryReason = boundaryReason,
            rawEvents = canonical.map { it.raw },
            cleanedEvents = cleaned,
            tokens = tokens,
            channels = channels,
            numeric = numericMap,
            nEventsFinal = n,
            nLoopRemoved = runKept.size - cleaned.size,
            nDedupRemoved = canonical.size - runKept.size,
            backRate = backRate,
            revisitRatio = revisit,
            spanSeconds = span
        )
    }

    private fun keepAfterRuns(events: List<CanonicalClick>): List<CanonicalClick> =
        events.filterIndexed { i, e -> i == 0 || e.eventToken != events[i - 1].eventToken }

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
                    repeats++
                    cursor += period
                }
                if (repeats >= 2) {
                    kept += events.subList(i, i + period)
                    i = cursor
                    matched = true
                    break
                }
            }
            if (!matched) kept += events[i++]
        }
        return kept
    }

    private fun normalizeEventType(value: String): String = when (value.trim().lowercase(Locale.US)) {
        "[cly]_view", "view" -> "view"
        "[cly]_action", "action" -> "action"
        "" -> "unknown"
        else -> value.trim().lowercase(Locale.US)
    }

    fun canonizeName(value: String): String {
        val text = value.trim()
        if (text.isEmpty() || text.lowercase(Locale.US) in setOf("nan", "none", "null", "<na>", "<missing>", "<none>")) {
            return MISSING
        }
        if (text.contains("://")) return canonizeUrl(text)
        return if ('/' in text) text.split('/').joinToString("/") { maskSegment(it) } else text
    }

    private fun canonizeUrl(value: String): String {
        val uri = runCatching { URI(value) }.getOrNull() ?: return value
        val host = uri.host ?: ""
        val path = (uri.path ?: "/").split('/').joinToString("/") { maskSegment(it) }.trimEnd('/').ifEmpty { "/" }
        val query = uri.rawQuery.orEmpty().split('&').mapNotNull { part ->
            val bits = part.split('=', limit = 2)
            if (bits.size != 2) return@mapNotNull null
            val key = URLDecoder.decode(bits[0], StandardCharsets.UTF_8.name()).lowercase(Locale.US)
            val decoded = URLDecoder.decode(bits[1], StandardCharsets.UTF_8.name())
            if (key in semanticQueryKeys && decoded.isNotBlank()) key to decoded else null
        }.sortedBy { it.first }.joinToString("&") { "${it.first}=${it.second}" }
        return host + path + if (query.isEmpty()) "" else "?$query"
    }

    private fun maskSegment(value: String): String = when {
        digitRegex.matches(value) -> "{id}"
        uuidRegex.matches(value) || hexRegex.matches(value) -> "{uuid}"
        codeRegex.matches(value) -> "{code}"
        else -> value
    }

    private fun quantile(values: List<Double>, q: Double): Double {
        if (values.isEmpty()) return 0.0
        val sorted = values.sorted()
        val position = (sorted.size - 1) * q
        val low = position.toInt()
        val high = kotlin.math.ceil(position).toInt()
        return if (low == high) sorted[low] else sorted[low] + (sorted[high] - sorted[low]) * (position - low)
    }
}
