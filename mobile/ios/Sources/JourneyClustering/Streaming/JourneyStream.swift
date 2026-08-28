import Foundation

/// Thread-safe, multi-session incremental journey stream manager for Swift/iOS.
/// Buffers events per session, detects journey boundaries, and runs scoring on completed boundaries.
public final class JourneyStream: @unchecked Sendable {
    private final class SessionState {
        var events: [ClickstreamEvent] = []
        var journeyNumber: Int = 1
    }

    private let model: JourneyModel
    private let outOfOrderToleranceMillis: Int64
    private let maxOpenSessions: Int
    private let maxBufferedEvents: Int
    private let lock = NSLock()
    private var states: [String: SessionState] = [:]
    private var sessionAccessOrder: [String] = []

    public init(
        model: JourneyModel,
        outOfOrderToleranceMillis: Int64 = 2_000,
        maxOpenSessions: Int = 128,
        maxBufferedEvents: Int = 1_000
    ) {
        self.model = model
        self.outOfOrderToleranceMillis = outOfOrderToleranceMillis
        self.maxOpenSessions = maxOpenSessions
        self.maxBufferedEvents = maxBufferedEvents
    }

    public func reset(sessionId: String? = nil) {
        lock.withLock {
            if let sid = sessionId {
                states.removeValue(forKey: sid)
                sessionAccessOrder.removeAll { $0 == sid }
            } else {
                states.removeAll()
                sessionAccessOrder.removeAll()
            }
        }
    }

    public func append(event: ClickstreamEvent, scoreProvisional: Bool = true) -> StreamUpdate {
        lock.lock()
        defer { lock.unlock() }

        if event.sessionId.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
            return StreamUpdate(finalized: [], provisional: nil, errors: ["missing_session_id"])
        }
        let eventTime = JourneyPreprocessor.eventTimeMillis(event)
        if eventTime <= 0 {
            return StreamUpdate(finalized: [], provisional: nil, errors: ["invalid_client_time"])
        }

        var evicted: [MobilePrediction] = []
        let sid = event.sessionId
        if states[sid] == nil && states.count >= maxOpenSessions {
            if let oldest = sessionAccessOrder.first, let oldestState = states[oldest] {
                if let pred = finalizeInternal(sessionId: oldest, state: oldestState, reason: "session_evicted") {
                    evicted.append(pred)
                }
                states.removeValue(forKey: oldest)
                sessionAccessOrder.removeFirst()
            }
        }

        let state: SessionState
        if let existing = states[sid] {
            state = existing
            sessionAccessOrder.removeAll { $0 == sid }
            sessionAccessOrder.append(sid)
        } else {
            state = SessionState()
            states[sid] = state
            sessionAccessOrder.append(sid)
        }

        let lastTime = state.events.last.map { JourneyPreprocessor.eventTimeMillis($0) }
        if let lt = lastTime, eventTime < lt - outOfOrderToleranceMillis {
            return StreamUpdate(finalized: evicted, provisional: nil, errors: ["event_too_late"])
        }

        let boundary: String?
        if lastTime == nil || eventTime < lastTime! {
            boundary = nil
        } else {
            boundary = model.preprocessor.boundaryBefore(currentEvents: state.events, nextEvent: event)
        }

        if let b = boundary {
            if let pred = finalizeInternal(sessionId: sid, state: state, reason: b) {
                evicted.append(pred)
            }
        }

        state.events.append(event)
        state.events.sort { a, b in
            let tA = JourneyPreprocessor.eventTimeMillis(a)
            let tB = JourneyPreprocessor.eventTimeMillis(b)
            if tA != tB { return tA < tB }
            return a.sourceIndex < b.sourceIndex
        }

        if state.events.count > maxBufferedEvents {
            if let pred = finalizeInternal(sessionId: sid, state: state, reason: "buffer_cap") {
                evicted.append(pred)
            }
        }

        let provisional: MobilePrediction?
        if scoreProvisional && !state.events.isEmpty {
            provisional = model.scoreEvents(
                journeyId: makeJourneyId(sessionId: sid, state: state),
                events: state.events,
                provisional: true
            )
        } else {
            provisional = nil
        }

        return StreamUpdate(finalized: evicted, provisional: provisional)
    }

    public func snapshotProvisional(sessionId: String? = nil) -> MobilePrediction? {
        lock.lock()
        defer { lock.unlock() }

        if let sid = sessionId {
            guard let state = states[sid] else { return nil }
            return model.scoreEvents(journeyId: makeJourneyId(sessionId: sid, state: state), events: state.events, provisional: true)
        } else if let firstEntry = states.first {
            return model.scoreEvents(journeyId: makeJourneyId(sessionId: firstEntry.key, state: firstEntry.value), events: firstEntry.value.events, provisional: true)
        }
        return nil
    }

    public func snapshotProvisionals() -> [MobilePrediction] {
        lock.lock()
        defer { lock.unlock() }

        return states.compactMap { (sid, state) in
            model.scoreEvents(journeyId: makeJourneyId(sessionId: sid, state: state), events: state.events, provisional: true)
        }
    }

    public func flush(sessionId: String, reason: String = "flush") -> MobilePrediction? {
        lock.lock()
        defer { lock.unlock() }

        guard let state = states[sessionId] else { return nil }
        let prediction = finalizeInternal(sessionId: sessionId, state: state, reason: reason)
        states.removeValue(forKey: sessionId)
        sessionAccessOrder.removeAll { $0 == sessionId }
        return prediction
    }

    public func flushAll(reason: String = "flush") -> [MobilePrediction] {
        lock.lock()
        defer { lock.unlock() }

        let keys = Array(states.keys)
        var results: [MobilePrediction] = []
        for sid in keys {
            if let state = states[sid] {
                if let pred = finalizeInternal(sessionId: sid, state: state, reason: reason) {
                    results.append(pred)
                }
                states.removeValue(forKey: sid)
            }
        }
        sessionAccessOrder.removeAll()
        return results
    }

    public func flushExpired(nowEpochMs: Int64) -> [MobilePrediction] {
        lock.lock()
        defer { lock.unlock() }

        let idleThreshold = Int64(model.preprocessor.idleGapSeconds * 1000.0)
        let expiredKeys = states.compactMap { (sid, state) -> String? in
            let last = state.events.last.map { JourneyPreprocessor.eventTimeMillis($0) } ?? nowEpochMs
            return (nowEpochMs - last > idleThreshold) ? sid : nil
        }

        var results: [MobilePrediction] = []
        for sid in expiredKeys {
            if let state = states[sid] {
                if let pred = finalizeInternal(sessionId: sid, state: state, reason: "idle_timeout") {
                    results.append(pred)
                }
                states.removeValue(forKey: sid)
                sessionAccessOrder.removeAll { $0 == sid }
            }
        }
        return results
    }

    private func finalizeInternal(sessionId: String, state: SessionState, reason: String) -> MobilePrediction? {
        guard !state.events.isEmpty else { return nil }
        let result = model.scoreEvents(
            journeyId: makeJourneyId(sessionId: sessionId, state: state),
            events: state.events,
            provisional: false,
            boundaryReason: reason
        )
        state.events.removeAll()
        state.journeyNumber += 1
        return result
    }

    private func makeJourneyId(sessionId: String, state: SessionState) -> String {
        let suffix = sessionId.count >= 8 ? String(sessionId.suffix(8)) : sessionId
        let jNum = String(format: "J%06d", state.journeyNumber)
        return "\(suffix)-\(jNum)"
    }
}
