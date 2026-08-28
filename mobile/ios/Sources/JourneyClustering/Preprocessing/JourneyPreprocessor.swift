import Foundation

/// Production clickstream preprocessing and journey segmentation for Swift/iOS.
/// Matches Python pipeline stages: canonize -> enrich -> tokenize -> segment -> postprocess.
public final class JourneyPreprocessor: @unchecked Sendable {
    public static let missing = "<missing>"
    public static let tokenSeparator = "@"

    private static let semanticQueryKeys: Set<String> = [
        "tab", "cat_id", "categoryid", "ordertype", "type", "step", "mode", "view", "status"
    ]
    private static let backMarkers: [String] = [
        "btn_back", "backbutton", "handleback", "nav_back", "/back", "goback", "close", "dismiss", "cancel"
    ]
    private static let authMarkers: [String] = [
        "continue_login", "login_with", "click_login", "log_out", "logout", "sign_out", "signin_success"
    ]
    private static let rootNames: Set<String> = [
        "HomeVC", "HomeGuestVC", "HOME", "android/Home", "guest/HOME"
    ]

    private static let uuidRegex = try! NSRegularExpression(
        pattern: "^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$",
        options: [.caseInsensitive]
    )
    private static let hexRegex = try! NSRegularExpression(pattern: "^[0-9a-f]{16,}$", options: [.caseInsensitive])
    private static let digitRegex = try! NSRegularExpression(pattern: "^\\d+$", options: [])
    private static let codeRegex = try! NSRegularExpression(pattern: "^(?=.*\\d)[A-Z0-9]{6,}$", options: [])

    public static let shared = JourneyPreprocessor()

    private let lock = NSLock()
    private var _idleGapSeconds: Double = 90.0
    private var _rootReturnMinEvents: Int = 6
    private var _maxJourneyLength: Int = 80
    private var _minJourneyLength: Int = 4
    private var _cutOnRootReturn: Bool = true
    private var _cutOnAuthChange: Bool = true

    public var idleGapSeconds: Double {
        get { lock.withLock { _idleGapSeconds } }
        set { lock.withLock { _idleGapSeconds = newValue } }
    }
    public var rootReturnMinEvents: Int {
        get { lock.withLock { _rootReturnMinEvents } }
        set { lock.withLock { _rootReturnMinEvents = newValue } }
    }
    public var maxJourneyLength: Int {
        get { lock.withLock { _maxJourneyLength } }
        set { lock.withLock { _maxJourneyLength = newValue } }
    }
    public var minJourneyLength: Int {
        get { lock.withLock { _minJourneyLength } }
        set { lock.withLock { _minJourneyLength = newValue } }
    }
    public var cutOnRootReturn: Bool {
        get { lock.withLock { _cutOnRootReturn } }
        set { lock.withLock { _cutOnRootReturn = newValue } }
    }
    public var cutOnAuthChange: Bool {
        get { lock.withLock { _cutOnAuthChange } }
        set { lock.withLock { _cutOnAuthChange = newValue } }
    }

    public init() {}

    public func configure(configJson: String) {
        guard let data = configJson.data(using: .utf8),
              let root = try? JSONSerialization.jsonObject(with: data) as? [String: Any] else {
            return
        }
        let segment = (root["segment"] as? [String: Any])
            ?? ((root["journey_contract"] as? [String: Any])?["segment"] as? [String: Any])
            ?? root

        lock.withLock {
            if let v = segment["idle_gap_seconds"] as? Double { _idleGapSeconds = v }
            if let v = segment["root_return_min_events"] as? Int { _rootReturnMinEvents = v }
            if let v = segment["max_journey_length"] as? Int { _maxJourneyLength = v }
            if let v = segment["min_journey_length"] as? Int { _minJourneyLength = v }
            if let v = segment["cut_on_root_return"] as? Bool { _cutOnRootReturn = v }
            if let v = segment["cut_on_auth_change"] as? Bool { _cutOnAuthChange = v }
        }
    }

    private static let isoWithMillis: ISO8601DateFormatter = {
        let f = ISO8601DateFormatter()
        f.formatOptions = [.withInternetDateTime, .withFractionalSeconds]
        return f
    }()

    private static let isoStandard: ISO8601DateFormatter = {
        let f = ISO8601DateFormatter()
        f.formatOptions = [.withInternetDateTime]
        return f
    }()

    public static func eventTimeMillis(_ event: ClickstreamEvent) -> Int64 {
        if event.timestamp > 0 { return event.timestamp }
        guard let text = event.createdAt?.trimmingCharacters(in: .whitespacesAndNewlines), !text.isEmpty else {
            return 0
        }
        if let direct = Int64(text) { return direct }
        if let date = isoWithMillis.date(from: text) ?? isoStandard.date(from: text) {
            return Int64(date.timeIntervalSince1970 * 1000.0)
        }
        return 0
    }

    /// Attach screen context, canonical paths, and multi-resolution semantic tokens.
    public func canonicalize(events: [ClickstreamEvent]) -> [CanonicalClick] {
        let ordered = events.sorted { a, b in
            let segA = a.segment.lowercased()
            let segB = b.segment.lowercased()
            if segA != segB { return segA < segB }
            if a.sessionId != b.sessionId { return a.sessionId < b.sessionId }
            let tA = JourneyPreprocessor.eventTimeMillis(a)
            let tB = JourneyPreprocessor.eventTimeMillis(b)
            if tA != tB { return tA < tB }
            return a.sourceIndex < b.sourceIndex
        }

        var result: [CanonicalClick] = []
        var previousStream: String? = nil
        var previousClick: CanonicalClick? = nil
        var currentScreenContext = JourneyPreprocessor.missing
        var sinceCut = 0

        for i in 0..<ordered.count {
            let raw = ordered[i]
            let time = JourneyPreprocessor.eventTimeMillis(raw)
            let stream = streamKey(raw)
            let sameStream = previousStream == stream
            let eventType = normalizeEventType(raw.key)
            let segmentName = canonizeName(raw.name)
            let nextRaw = (i + 1 < ordered.count) ? ordered[i + 1] : nil
            let sameNext = nextRaw != nil && streamKey(nextRaw!) == stream

            let gapPrev: Double? = (sameStream && previousClick != nil)
                ? Double(time - previousClick!.eventTimeMillis) / 1000.0
                : nil
            let gapNext: Double? = (sameNext && nextRaw != nil)
                ? Double(JourneyPreprocessor.eventTimeMillis(nextRaw!) - time) / 1000.0
                : nil

            // Boundary check for screen context reset
            let boundary: String? = (sameStream && previousClick != nil)
                ? boundaryBeforeCanonical(
                    previous: previousClick!,
                    currentEventType: eventType,
                    currentSegmentName: segmentName,
                    gapPrevSeconds: gapPrev,
                    sinceCut: sinceCut
                )
                : nil

            if !sameStream || boundary != nil {
                currentScreenContext = JourneyPreprocessor.missing
                sinceCut = 0
            }

            let explicitTrimmed = raw.screenId.trimmingCharacters(in: .whitespacesAndNewlines)
            let explicitScreen: String?
            if !explicitTrimmed.isEmpty && !["nan", "none", "null", "<missing>"].contains(explicitTrimmed.lowercased()) {
                explicitScreen = canonizeName(explicitTrimmed)
            } else {
                explicitScreen = nil
            }

            let screen: String
            if let exp = explicitScreen {
                screen = exp
            } else if eventType == "view" {
                screen = segmentName
            } else {
                screen = currentScreenContext
            }

            let eventToken: String
            if eventType == "view" {
                eventToken = "view\(JourneyPreprocessor.tokenSeparator)\(screen)"
            } else {
                eventToken = "action\(JourneyPreprocessor.tokenSeparator)\(screen)#\(segmentName)"
            }

            // Semantic enrichment
            let semantics = SemanticTaxonomy.classifyEvent(eventType: eventType, screenContext: screen, segmentName: segmentName)
            let tokenL1 = semantics.businessFamily
            let tokenL2 = "\(semantics.businessFamily)/\(semantics.businessModule)"
            let tokenL3 = "\(semantics.businessFamily)/\(semantics.businessModule)/\(semantics.businessObject)/\(semantics.operation)"
            let operationToken = "\(semantics.operationStage):\(semantics.operation)"

            let canonical = CanonicalClick(
                raw: raw,
                eventType: eventType,
                segmentName: segmentName,
                screenContext: screen,
                eventToken: eventToken,
                eventTimeMillis: time,
                gapPrevSeconds: gapPrev,
                gapNextSeconds: gapNext,
                semantics: semantics,
                tokenL1: tokenL1,
                tokenL2: tokenL2,
                tokenL3: tokenL3,
                operationToken: operationToken
            )

            result.append(canonical)
            if eventType == "view" {
                currentScreenContext = screen
            }
            previousStream = stream
            previousClick = canonical
            sinceCut += 1
        }
        return result
    }

    private func streamKey(_ event: ClickstreamEvent) -> String {
        return "\(event.segment.trimmingCharacters(in: .whitespacesAndNewlines).lowercased())\u{1F}\(event.sessionId)"
    }

    public func prepareAll(events: [ClickstreamEvent]) -> [PreparedJourney] {
        let canonical = canonicalize(events: events)
        guard !canonical.isEmpty else { return [] }
        var output: [PreparedJourney] = []
        var start = 0
        var startReason = "session_start"

        for index in 1..<canonical.count {
            let prev = canonical[index - 1]
            let curr = canonical[index]
            let boundary = boundaryBetween(previous: prev, current: curr, sinceCut: index - start)
            if let b = boundary {
                let jId = String(format: "J%06d", output.count + 1)
                let slice = Array(canonical[start..<index])
                output.append(buildJourney(journeyId: jId, canonical: slice, boundaryReason: startReason))
                start = index
                startReason = b
            }
        }

        let jId = String(format: "J%06d", output.count + 1)
        let slice = Array(canonical[start..<canonical.count])
        output.append(buildJourney(journeyId: jId, canonical: slice, boundaryReason: startReason))
        return output
    }

    public func prepareSingle(journeyId: String, events: [ClickstreamEvent], boundaryReason: String = "") -> PreparedJourney? {
        let canonical = canonicalize(events: events)
        guard !canonical.isEmpty else { return nil }
        return buildJourney(journeyId: journeyId, canonical: canonical, boundaryReason: boundaryReason)
    }

    public func boundaryBefore(currentEvents: [ClickstreamEvent], nextEvent: ClickstreamEvent) -> String? {
        guard let previousRaw = currentEvents.last else { return nil }
        if previousRaw.sessionId != nextEvent.sessionId ||
            previousRaw.segment.caseInsensitiveCompare(nextEvent.segment) != .orderedSame {
            return "session_start"
        }

        let pair = canonicalize(events: [previousRaw, nextEvent])
        guard pair.count == 2, pair.last?.raw == nextEvent else { return nil }
        return boundaryBetween(previous: pair[0], current: pair[1], sinceCut: currentEvents.count)
    }

    private func boundaryBetween(previous: CanonicalClick, current: CanonicalClick, sinceCut: Int) -> String? {
        let idleGap = self.idleGapSeconds
        let maxLen = self.maxJourneyLength
        let rootEvents = self.rootReturnMinEvents
        let cutRoot = self.cutOnRootReturn
        let cutAuth = self.cutOnAuthChange

        if current.raw.sessionId != previous.raw.sessionId ||
            current.raw.segment.caseInsensitiveCompare(previous.raw.segment) != .orderedSame {
            return "session_start"
        }
        if (current.gapPrevSeconds ?? 0.0) > idleGap {
            return "idle_gap"
        }
        if sinceCut >= maxLen {
            return "length_cap"
        }
        if cutRoot && current.eventType == "view" &&
            JourneyPreprocessor.rootNames.contains(current.segmentName) &&
            !JourneyPreprocessor.rootNames.contains(previous.segmentName) &&
            sinceCut >= rootEvents {
            return "root_return"
        }
        if cutAuth && isAuthAction(eventType: current.eventType, name: current.segmentName) &&
            !isAuthAction(eventType: previous.eventType, name: previous.segmentName) {
            return "auth_change"
        }
        return nil
    }

    private func boundaryBeforeCanonical(
        previous: CanonicalClick,
        currentEventType: String,
        currentSegmentName: String,
        gapPrevSeconds: Double?,
        sinceCut: Int
    ) -> String? {
        let idleGap = self.idleGapSeconds
        let maxLen = self.maxJourneyLength
        let rootEvents = self.rootReturnMinEvents
        let cutRoot = self.cutOnRootReturn
        let cutAuth = self.cutOnAuthChange

        if (gapPrevSeconds ?? 0.0) > idleGap {
            return "idle_gap"
        }
        if sinceCut >= maxLen {
            return "length_cap"
        }
        if cutRoot && currentEventType == "view" &&
            JourneyPreprocessor.rootNames.contains(currentSegmentName) &&
            !JourneyPreprocessor.rootNames.contains(previous.segmentName) &&
            sinceCut >= rootEvents {
            return "root_return"
        }
        if cutAuth && isAuthAction(eventType: currentEventType, name: currentSegmentName) &&
            !isAuthAction(eventType: previous.eventType, name: previous.segmentName) {
            return "auth_change"
        }
        return nil
    }

    private func isAuthAction(eventType: String, name: String) -> Bool {
        guard eventType == "action" else { return false }
        let low = name.lowercased()
        return JourneyPreprocessor.authMarkers.contains { low.contains($0) }
    }

    private func buildJourney(journeyId: String, canonical: [CanonicalClick], boundaryReason: String) -> PreparedJourney {
        let runKept = keepAfterRuns(canonical)
        let cleaned = keepAfterCycles(runKept)
        let tokens = cleaned.map { $0.eventToken }
        let coarseTokens = cleaned.map { $0.tokenL2 }
        let intentTokens = cleaned.map { $0.tokenL3 }
        let operationTokens = cleaned.map { $0.operationToken }

        let n = tokens.count
        let gaps = cleaned.dropFirst().compactMap { $0.gapPrevSeconds }.filter { $0 >= 0.0 }
        let span: Double
        if canonical.count > 1 {
            span = max(0.0, Double(canonical.last!.eventTimeMillis - canonical.first!.eventTimeMillis) / 1000.0)
        } else {
            span = 0.0
        }
        let unique = Set(tokens).count
        let backRate: Double
        if n == 0 {
            backRate = 0.0
        } else {
            let backCount = cleaned.filter { click in
                let low = click.segmentName.lowercased()
                return JourneyPreprocessor.backMarkers.contains { low.contains($0) }
            }.count
            backRate = Double(backCount) / Double(n)
        }
        let revisit = (n == 0) ? 0.0 : (1.0 - Double(unique) / Double(n))
        let actionRatio = (n == 0) ? 0.0 : Double(cleaned.filter { $0.eventType == "action" }.count) / Double(n)

        let numericMap: [String: Double] = [
            "n_events_final": Double(n),
            "n_unique_tokens": Double(unique),
            "action_ratio": actionRatio,
            "back_rate": backRate,
            "revisit_ratio": revisit,
            "n_loop_removed": Double(runKept.count - cleaned.count),
            "n_dedup_removed": Double(canonical.count - runKept.count),
            "span_seconds": span,
            "median_gap_s": quantile(gaps, q: 0.5),
            "p90_gap_s": quantile(gaps, q: 0.9),
            "max_gap_s": gaps.max() ?? 0.0
        ]

        let channels: [String: [String]] = [
            "primary": tokens,
            "coarse": coarseTokens,
            "intent": intentTokens,
            "operation": operationTokens
        ]

        return PreparedJourney(
            journeyId: journeyId,
            sessionId: canonical.first!.raw.sessionId,
            deviceId: canonical.first!.raw.deviceId,
            customerId: canonical.first!.raw.customerId,
            platform: canonical.first!.raw.segment.lowercased(),
            boundaryReason: boundaryReason,
            rawEvents: canonical.map { $0.raw },
            cleanedEvents: cleaned,
            tokens: tokens,
            channels: channels,
            numeric: numericMap,
            nEventsFinal: n,
            nLoopRemoved: runKept.count - cleaned.count,
            nDedupRemoved: canonical.count - runKept.count,
            backRate: backRate,
            revisitRatio: revisit,
            spanSeconds: span
        )
    }

    public func keepAfterRuns(_ events: [CanonicalClick]) -> [CanonicalClick] {
        return events.enumerated().compactMap { i, e in
            (i == 0 || e.eventToken != events[i - 1].eventToken) ? e : nil
        }
    }

    public func keepAfterCycles(_ events: [CanonicalClick]) -> [CanonicalClick] {
        var kept: [CanonicalClick] = []
        var i = 0
        while i < events.count {
            var matched = false
            for period in 2...4 {
                if i + period * 2 > events.count { continue }
                let block = Array(events[i..<(i + period)]).map { $0.eventToken }
                var repeats = 1
                var cursor = i + period
                while cursor + period <= events.count {
                    let nextBlock = Array(events[cursor..<(cursor + period)]).map { $0.eventToken }
                    if nextBlock == block {
                        repeats += 1
                        cursor += period
                    } else {
                        break
                    }
                }
                if repeats >= 2 {
                    kept.append(contentsOf: events[i..<(i + period)])
                    i = cursor
                    matched = true
                    break
                }
            }
            if !matched {
                kept.append(events[i])
                i += 1
            }
        }
        return kept
    }

    private func normalizeEventType(_ value: String) -> String {
        let low = value.trimmingCharacters(in: .whitespacesAndNewlines).lowercased()
        switch low {
        case "[cly]_view", "view":
            return "view"
        case "[cly]_action", "action":
            return "action"
        case "":
            return "unknown"
        default:
            return low
        }
    }

    public func canonizeName(_ value: String) -> String {
        let text = value.trimmingCharacters(in: .whitespacesAndNewlines)
        if text.isEmpty || ["nan", "none", "null", "<na>", "<missing>", "<none>"].contains(text.lowercased()) {
            return JourneyPreprocessor.missing
        }
        if text.contains("://") {
            return canonizeUrl(text)
        }
        if text.contains("/") {
            let segments = text.components(separatedBy: "/")
            return segments.map { maskSegment($0) }.joined(separator: "/")
        }
        return text
    }

    private func canonizeUrl(_ value: String) -> String {
        guard let url = URL(string: value) else { return value }
        let host = url.host ?? ""
        let rawPath = url.path.isEmpty ? "/" : url.path
        let pathSegs = rawPath.components(separatedBy: "/").map { maskSegment($0) }
        var path = pathSegs.joined(separator: "/")
        while path.count > 1 && path.hasSuffix("/") {
            path.removeLast()
        }
        if path.isEmpty { path = "/" }

        var query = ""
        if let queryItems = URLComponents(string: value)?.queryItems {
            let filtered = queryItems.compactMap { item -> (String, String)? in
                let k = item.name.lowercased()
                let v = item.value ?? ""
                if JourneyPreprocessor.semanticQueryKeys.contains(k) && !v.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
                    return (k, v)
                }
                return nil
            }.sorted { $0.0 < $1.0 }

            query = filtered.map { "\($0.0)=\($0.1)" }.joined(separator: "&")
        }

        return host + path + (query.isEmpty ? "" : "?\(query)")
    }

    private func maskSegment(_ value: String) -> String {
        let range = NSRange(location: 0, length: value.utf16.count)
        if JourneyPreprocessor.digitRegex.firstMatch(in: value, options: [], range: range) != nil {
            return "{id}"
        }
        if JourneyPreprocessor.uuidRegex.firstMatch(in: value, options: [], range: range) != nil ||
            JourneyPreprocessor.hexRegex.firstMatch(in: value, options: [], range: range) != nil {
            return "{uuid}"
        }
        if JourneyPreprocessor.codeRegex.firstMatch(in: value, options: [], range: range) != nil {
            return "{code}"
        }
        return value
    }

    private func quantile(_ values: [Double], q: Double) -> Double {
        guard !values.isEmpty else { return 0.0 }
        let sorted = values.sorted()
        let position = Double(sorted.count - 1) * q
        let low = Int(floor(position))
        let high = Int(ceil(position))
        if low == high { return sorted[low] }
        return sorted[low] + (sorted[high] - sorted[low]) * (position - Double(low))
    }
}
