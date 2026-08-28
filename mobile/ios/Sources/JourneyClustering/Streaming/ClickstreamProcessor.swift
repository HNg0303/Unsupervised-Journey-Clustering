import Foundation

/// Production clickstream processor facade supporting FIXED_DURATION and JOURNEY_COMPLETE routes.
/// Receives live clickstream events or raw JSON payloads and dispatches updates.
public final class ClickstreamProcessor: @unchecked Sendable {
    public let model: JourneyModel
    public let route: MobileProcessingRoute
    public let windowDurationSeconds: Int64

    private let stream: JourneyStream
    private let lock = NSLock()
    private var windowStartIndex: Int = 1
    private var windowStartTimestamp: Int64? = nil
    private var windowEventsCount: Int = 0
    private var finalizedInWindow: [MobilePrediction] = []

    public init(
        model: JourneyModel,
        route: MobileProcessingRoute = .journeyComplete,
        windowDurationSeconds: Int64 = 30
    ) {
        self.model = model
        self.route = route
        self.windowDurationSeconds = windowDurationSeconds
        self.stream = JourneyStream(model: model)
    }

    public func start() {
        lock.withLock {
            stream.reset()
            windowStartIndex = 1
            windowStartTimestamp = nil
            windowEventsCount = 0
            finalizedInWindow.removeAll()
        }
    }

    public func accept(event: ClickstreamEvent) -> MobileProcessingUpdate {
        lock.lock()
        defer { lock.unlock() }

        let eventTime = JourneyPreprocessor.eventTimeMillis(event)
        if windowStartTimestamp == nil && eventTime > 0 {
            windowStartTimestamp = eventTime
        }

        let update = stream.append(event: event, scoreProvisional: (route == .fixedDuration))
        if route == .fixedDuration {
            finalizedInWindow.append(contentsOf: update.finalized)
        }
        windowEventsCount += 1

        if route == .journeyComplete {
            return MobileProcessingUpdate(
                route: route,
                windowIndex: nil,
                eventsInWindow: 1,
                windowStartTimestamp: eventTime,
                windowEndTimestamp: eventTime,
                finalized: update.finalized,
                provisional: [],
                errors: update.errors
            )
        }

        // FIXED_DURATION route
        let start = windowStartTimestamp ?? eventTime
        if eventTime - start >= windowDurationSeconds * 1000 {
            let emittedFinalized = finalizedInWindow
            finalizedInWindow.removeAll()
            let emittedProvisional = stream.snapshotProvisionals()
            let wIndex = windowStartIndex
            windowStartIndex += 1
            let wStart = start
            windowStartTimestamp = eventTime
            let count = windowEventsCount
            windowEventsCount = 0

            return MobileProcessingUpdate(
                route: route,
                windowIndex: wIndex,
                eventsInWindow: count,
                windowStartTimestamp: wStart,
                windowEndTimestamp: eventTime,
                finalized: emittedFinalized,
                provisional: emittedProvisional,
                errors: update.errors
            )
        }

        return MobileProcessingUpdate(
            route: route,
            windowIndex: nil,
            eventsInWindow: windowEventsCount,
            windowStartTimestamp: start,
            windowEndTimestamp: eventTime,
            finalized: [],
            provisional: [],
            errors: update.errors
        )
    }

    public func acceptJson(rawJson: String) -> [MobileProcessingUpdate] {
        let trimmed = rawJson.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty else { return [] }
        let events = ClickstreamRepository.parseJsonEvents(trimmed)
        return events.map { accept(event: $0) }.filter { $0.emitted }
    }

    public func finish(sessionId: String? = nil) -> MobileProcessingUpdate? {
        lock.lock()
        defer { lock.unlock() }

        let flushed: [MobilePrediction]
        if let sid = sessionId {
            flushed = stream.flush(sessionId: sid, reason: "session_end").map { [$0] } ?? []
        } else {
            flushed = stream.flushAll(reason: "stream_finished")
        }

        finalizedInWindow.append(contentsOf: flushed)
        if finalizedInWindow.isEmpty && windowEventsCount == 0 { return nil }

        let emittedFinalized = finalizedInWindow
        finalizedInWindow.removeAll()

        let now = Int64(Date().timeIntervalSince1970 * 1000.0)
        let wIndex = (route == .fixedDuration) ? windowStartIndex : nil
        if route == .fixedDuration { windowStartIndex += 1 }

        return MobileProcessingUpdate(
            route: route,
            windowIndex: wIndex,
            eventsInWindow: windowEventsCount,
            windowStartTimestamp: windowStartTimestamp,
            windowEndTimestamp: now,
            finalized: emittedFinalized,
            provisional: [],
            errors: []
        )
    }
}
