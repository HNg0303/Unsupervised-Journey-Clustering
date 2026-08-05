package vn.hifpt.clickstream

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
    val screenBare: String,
    val screen: String,
    val target: String,
    val screenClass: String,
    val isBack: Boolean,
    val durationClip: Double,
    val eventTimeMillis: Long,
    val gapSeconds: Double,
    val tokenL2: String
)

/** Kotlin implementation of canonize -> segment -> postprocess used by the fitted scorer. */
object MobileJourneyPreprocessor {
    private const val MISSING = "<none>"
    private const val MID_PATH_DEPTH = 3
    private const val IDLE_GAP_SECONDS = 90.0
    private const val ROOT_RETURN_MIN_EVENTS = 6
    private const val MAX_JOURNEY_LENGTH = 80

    private val chromeScreens = setOf(
        "MainTabBarController", "BaseNavigation", "UINavigationController", "UIViewController",
        "UITrackingElementWindowController", "_UISceneHostingViewController",
        "UIHostingController<PopupView>", "SFBrowserRemoteViewController",
        "SFAuthenticationViewController", "SFSafariViewController", "WebkitEcommerceController",
        "HiWebViewActivity", "WebViewActivity"
    )
    private val bootScreens = setOf("SplashVC", "SplashActivity", "Splash", "MainAppActivity", "LaunchScreen")
    private val rootScreens = setOf("HomeVC", "HomeGuestVC", "HOME", "android/Home", "guest/HOME")
    private val semanticQueryKeys = setOf("tab", "cat_id", "categoryid", "ordertype", "type", "step", "mode", "view", "status")
    private val backMarkers = listOf("btn_back", "backbutton", "handleback", "/back", "goback", "close", "dismiss", "cancel")
    private val authMarkers = listOf("continue_login", "login_with", "click_login", "log_out", "logout", "sign_out", "signin_success")
    private val uuidRegex = Regex("^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$", RegexOption.IGNORE_CASE)
    private val hexRegex = Regex("^[0-9a-f]{16,}$", RegexOption.IGNORE_CASE)
    private val digitRegex = Regex("^\\d+$")
    private val codeRegex = Regex("^(?=.*\\d)[A-Z0-9]{6,}$")

    fun prepareAll(events: List<ClickstreamEvent>): List<PreparedJourney> {
        val canonical = canonicalize(events)
        if (canonical.isEmpty()) return emptyList()
        val slices = mutableListOf<JourneySlice>()
        var start = 0
        var reason = "session_start"

        for (index in 1 until canonical.size) {
            val previous = canonical[index - 1]
            val current = canonical[index]
            val sinceCut = index - start
            val boundary = when {
                current.raw.sessionId != previous.raw.sessionId -> "session_start"
                current.gapSeconds > IDLE_GAP_SECONDS -> "idle_gap"
                sinceCut >= MAX_JOURNEY_LENGTH -> "length_cap"
                current.screenBare in rootScreens && previous.screenBare !in rootScreens &&
                    sinceCut >= ROOT_RETURN_MIN_EVENTS -> "root_return"
                isAuthAction(current) && !isAuthAction(previous) -> "auth_change"
                else -> null
            }
            if (boundary != null) {
                slices += JourneySlice(start, index, reason)
                start = index
                reason = boundary
            }
        }
        slices += JourneySlice(start, canonical.size, reason)

        return slices.mapIndexed { index, slice ->
            buildJourney(
                journeyId = "J${(index + 1).toString().padStart(6, '0')}",
                canonical = canonical.subList(slice.start, slice.end),
                boundaryReason = slice.reason
            )
        }
    }

    fun prepareSingle(
        journeyId: String,
        events: List<ClickstreamEvent>,
        boundaryReason: String = ""
    ): PreparedJourney? {
        val canonical = canonicalize(events)
        if (canonical.isEmpty()) return null
        return buildJourney(journeyId, canonical, boundaryReason)
    }

    /** Return the boundary reason that should be applied before appending nextEvent. */
    fun boundaryBefore(currentEvents: List<ClickstreamEvent>, nextEvent: ClickstreamEvent): String? {
        if (currentEvents.isEmpty()) return null
        if (nextEvent.sessionId != currentEvents.last().sessionId) return "session_start"
        val ordered = canonicalize(listOf(currentEvents.last(), nextEvent))
        if (ordered.size < 2) return null
        val previous = ordered[ordered.lastIndex - 1]
        val current = ordered.last()
        val sinceCut = currentEvents.size
        return when {
            current.raw.sessionId != previous.raw.sessionId -> "session_start"
            current.gapSeconds > IDLE_GAP_SECONDS -> "idle_gap"
            sinceCut >= MAX_JOURNEY_LENGTH -> "length_cap"
            current.screenBare in rootScreens && previous.screenBare !in rootScreens &&
                sinceCut >= ROOT_RETURN_MIN_EVENTS -> "root_return"
            isAuthAction(current) && !isAuthAction(previous) -> "auth_change"
            else -> null
        }
    }

    private fun canonicalize(events: List<ClickstreamEvent>): List<CanonicalClick> {
        val ordered = events.sortedWith(
            compareBy<ClickstreamEvent> { it.sessionId }
                .thenBy { eventTime(it) }
                .thenBy { it.sourceIndex }
        )
        var previousSession = ""
        var previousTime = 0L
        return ordered.map { raw ->
            val eventTime = eventTime(raw)
            val eventType = when {
                raw.key.equals("[CLY]_view", ignoreCase = true) -> "View"
                raw.key.equals("action", ignoreCase = true) -> "Action"
                else -> raw.key.trim().replaceFirstChar { it.titlecase(Locale.US) }
            }
            val screenRaw = if (eventType == "View") raw.name else raw.screenId
            val targetRaw = if (eventType == "View") MISSING else raw.name
            val screenBare = canonizeName(screenRaw)
            val target = canonizeName(targetRaw)
            val platform = raw.segment.ifBlank { "unk" }
            val screen = "$platform::$screenBare"
            val tokenL1 = "$eventType@$screen"
            val tokenL2 = if (target == MISSING) tokenL1 else "$tokenL1#${shallowPath(target)}"
            val gap = if (raw.sessionId == previousSession && previousTime > 0L) {
                max(0.0, (eventTime - previousTime) / 1000.0)
            } else {
                0.0
            }
            previousSession = raw.sessionId
            previousTime = eventTime
            CanonicalClick(
                raw = raw,
                eventType = eventType,
                screenBare = screenBare,
                screen = screen,
                target = target,
                screenClass = when {
                    screenBare in bootScreens -> "boot"
                    screenBare in chromeScreens -> "chrome"
                    else -> "ux"
                },
                isBack = backMarkers.any { target.lowercase(Locale.US).contains(it) },
                durationClip = raw.durationSeconds.toDouble().coerceIn(0.0, 1800.0),
                eventTimeMillis = eventTime,
                gapSeconds = gap,
                tokenL2 = tokenL2
            )
        }
    }

    private fun buildJourney(
        journeyId: String,
        canonical: List<CanonicalClick>,
        boundaryReason: String
    ): PreparedJourney {
        val filtered = canonical.filter { it.screenClass != "chrome" && it.screenClass != "boot" }
        val runKept = keepAfterRuns(filtered)
        val cycleKept = keepAfterCycles(runKept)
        val cleaned = cycleKept
        val tokens = cleaned.map { it.tokenL2 }
        val n = tokens.size
        val actions = cleaned.count { it.eventType == "Action" }
        val unique = tokens.toSet().size
        val gaps = cleaned.drop(1).map { it.gapSeconds }
        val dwell = cleaned.filter { it.eventType != "Action" }.sumOf { it.durationClip }
        val span = if (canonical.size > 1) {
            max(0.0, (canonical.last().eventTimeMillis - canonical.first().eventTimeMillis) / 1000.0)
        } else {
            0.0
        }
        val medianGap = median(gaps)
        val backRate = if (n == 0) 0.0 else cleaned.count { it.isBack }.toDouble() / n
        val revisit = if (n == 0) 0.0 else 1.0 - unique.toDouble() / n

        return PreparedJourney(
            journeyId = journeyId,
            sessionId = canonical.first().raw.sessionId,
            deviceId = canonical.first().raw.deviceId,
            customerId = canonical.first().raw.customerId,
            platform = canonical.first().raw.segment,
            boundaryReason = boundaryReason,
            rawEvents = canonical.map { it.raw },
            cleanedEvents = cleaned,
            tokens = tokens,
            numeric = mapOf(
                "n_events_final" to n.toDouble(),
                "n_unique_tokens" to unique.toDouble(),
                "action_ratio" to if (n == 0) 0.0 else actions.toDouble() / n,
                "back_rate" to backRate,
                "revisit_ratio" to revisit,
                "n_loop_removed" to (runKept.size - cycleKept.size).toDouble(),
                "n_dedup_removed" to (filtered.size - runKept.size).toDouble(),
                "span_seconds" to span,
                "total_dwell_s" to dwell,
                "median_gap_s" to medianGap
            ),
            nEventsFinal = n,
            nLoopRemoved = runKept.size - cycleKept.size,
            nDedupRemoved = filtered.size - runKept.size,
            backRate = backRate,
            revisitRatio = revisit,
            spanSeconds = span
        )
    }

    private fun keepAfterRuns(events: List<CanonicalClick>): List<CanonicalClick> {
        if (events.isEmpty()) return emptyList()
        return events.filterIndexed { index, event -> index == 0 || event.tokenL2 != events[index - 1].tokenL2 }
    }

    private fun keepAfterCycles(events: List<CanonicalClick>): List<CanonicalClick> {
        val kept = mutableListOf<CanonicalClick>()
        var index = 0
        while (index < events.size) {
            var matched = false
            for (period in 2..4) {
                if (index + period * 2 > events.size) continue
                val block = events.subList(index, index + period).map { it.tokenL2 }
                var repeats = 1
                var cursor = index + period
                while (cursor + period <= events.size &&
                    events.subList(cursor, cursor + period).map { it.tokenL2 } == block
                ) {
                    repeats += 1
                    cursor += period
                }
                if (repeats >= 2) {
                    kept += events.subList(index, index + period)
                    index = cursor
                    matched = true
                    break
                }
            }
            if (!matched) {
                kept += events[index]
                index += 1
            }
        }
        return kept
    }

    private fun isAuthAction(event: CanonicalClick): Boolean =
        event.eventType == "Action" && authMarkers.any { event.target.lowercase(Locale.US).contains(it) }

    private fun shallowPath(target: String): String {
        if (target == MISSING) return MISSING
        val parts = target.split('/').filter { it.isNotEmpty() }
        return if (parts.size <= MID_PATH_DEPTH) parts.joinToString("/")
        else parts.take(MID_PATH_DEPTH).joinToString("/") + "/*"
    }

    private fun canonizeName(value: String): String {
        val text = value.trim()
        if (text.isEmpty() || text.lowercase(Locale.US) in setOf("nan", "none", "null")) return MISSING
        if (text.contains("://")) return canonizeUrl(text)
        return if ('/' in text) text.split('/').joinToString("/") { maskSegment(it) } else text
    }

    private fun canonizeUrl(value: String): String {
        val uri = runCatching { URI(value) }.getOrNull() ?: return canonizeNameWithoutUrl(value)
        val host = uri.host ?: ""
        val path = (uri.path ?: "/").split('/').joinToString("/") { maskSegment(it) }
            .trimEnd('/').ifEmpty { "/" }
        val query = uri.rawQuery.orEmpty().split('&')
            .mapNotNull { part ->
                val bits = part.split('=', limit = 2)
                if (bits.size != 2) return@mapNotNull null
                val key = URLDecoder.decode(bits[0], StandardCharsets.UTF_8.name()).lowercase(Locale.US)
                if (key !in semanticQueryKeys) return@mapNotNull null
                val queryValue = URLDecoder.decode(bits[1], StandardCharsets.UTF_8.name())
                if (queryValue.isBlank()) null else key to queryValue
            }
            .sortedBy { it.first }
            .joinToString("&") { "${it.first}=${it.second}" }
        return host + path + if (query.isBlank()) "" else "?$query"
    }

    private fun canonizeNameWithoutUrl(value: String): String =
        if ('/' in value) value.split('/').joinToString("/") { maskSegment(it) } else value

    private fun maskSegment(value: String): String = when {
        value.isEmpty() -> value
        digitRegex.matches(value) -> "{id}"
        uuidRegex.matches(value) -> "{uuid}"
        hexRegex.matches(value) -> "{uuid}"
        codeRegex.matches(value) -> "{code}"
        else -> value
    }

    private fun eventTime(event: ClickstreamEvent): Long =
        event.createdAt?.let { runCatching { Instant.parse(it).toEpochMilli() }.getOrNull() }
            ?: event.timestamp

    private fun median(values: List<Double>): Double {
        if (values.isEmpty()) return 0.0
        val sorted = values.sorted()
        val middle = sorted.size / 2
        return if (sorted.size % 2 == 0) (sorted[middle - 1] + sorted[middle]) / 2.0 else sorted[middle]
    }

    private data class JourneySlice(val start: Int, val end: Int, val reason: String)
}
