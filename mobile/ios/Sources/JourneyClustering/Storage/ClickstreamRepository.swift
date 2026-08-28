import Foundation

/// Repository for parsing production clickstream CSV logs and JSON payloads into ClickstreamEvent models.
public final class ClickstreamRepository: @unchecked Sendable {
    public init() {}

    /// Parse CSV or JSON content into replay sessions grouped by sessionId.
    public static func loadSessions(content: String, isCsv: Bool) -> [ReplaySession] {
        let events = isCsv ? parseCsvEvents(content) : parseJsonEvents(content)
        let filtered = events.filter { !$0.sessionId.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty }

        let grouped = Dictionary(grouping: filtered, by: { $0.sessionId })
        return grouped.map { (sessionId, sessionEvents) in
            let sorted = sessionEvents.sorted { a, b in
                if a.timestamp != b.timestamp { return a.timestamp < b.timestamp }
                return a.sourceIndex < b.sourceIndex
            }
            return ReplaySession(id: sessionId, events: sorted)
        }.sorted { $0.events.count > $1.events.count }
    }

    /// Parse production CSV schema used by mobile clickstream dumps.
    public static func parseCsvEvents(_ csv: String) -> [ClickstreamEvent] {
        let rows = parseCsvRows(csv)
        guard !rows.isEmpty else { return [] }

        let header = rows[0].map { $0.trimmingCharacters(in: .whitespacesAndNewlines).replacingOccurrences(of: "\u{FEFF}", with: "") }
        var indexMap: [String: Int] = [:]
        for (idx, name) in header.enumerated() {
            indexMap[name] = idx
        }

        func val(_ row: [String], _ names: [String]) -> String {
            for name in names {
                if let idx = indexMap[name], idx < row.count {
                    let v = row[idx].trimmingCharacters(in: .whitespacesAndNewlines)
                    if !v.isEmpty { return v }
                }
            }
            return ""
        }

        var events: [ClickstreamEvent] = []
        for (sequence, row) in rows.dropFirst().enumerated() {
            if row.allSatisfy({ $0.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty }) {
                continue
            }
            let clientTime = val(row, ["client_time", "timestamp"])
            let timestamp = parseClientTimeToMillis(clientTime)
            let sessionId = val(row, ["session_id"])
            if sessionId.isEmpty { continue }

            let customerIdRaw = val(row, ["customer_id"])
            let customerId: String?
            if !customerIdRaw.isEmpty && !["null", "none", "nan"].contains(customerIdRaw.lowercased()) {
                if customerIdRaw.hasSuffix(".0") && customerIdRaw.dropLast(2).allSatisfy({ $0.isNumber }) {
                    customerId = String(customerIdRaw.dropLast(2))
                } else {
                    customerId = customerIdRaw
                }
            } else {
                customerId = nil
            }

            let createdAtRaw = val(row, ["device_created_at", "created_at"])
            let createdAt = !createdAtRaw.isEmpty ? createdAtRaw : clientTime

            let segment = val(row, ["platform", "segmentation_segment", "segment"])
            let name = val(row, ["segmentation_name", "name"])
            let key = val(row, ["key"])

            let event = ClickstreamEvent(
                sourceIndex: sequence,
                eventId: val(row, ["_id", "event_id"]).isEmpty ? "row-\(sequence)" : val(row, ["_id", "event_id"]),
                deviceId: val(row, ["device_id"]),
                customerId: customerId,
                sessionId: sessionId,
                createdAt: createdAt.isEmpty ? nil : createdAt,
                timestamp: timestamp,
                key: key.isEmpty ? "unknown" : key,
                segment: segment.isEmpty ? "iOS" : segment,
                name: name.isEmpty ? (key.isEmpty ? "unknown" : key) : name,
                screenId: val(row, ["screen_id", "segmentation_screen_id"]),
                durationSeconds: Int(Double(val(row, ["dur", "duration"])) ?? 0.0),
                visit: val(row, ["visit"]).isEmpty ? nil : val(row, ["visit"])
            )
            events.append(event)
        }
        return events
    }

    /// Parse JSON arrays or single objects from live SDK streams or test files.
    public static func parseJsonEvents(_ json: String) -> [ClickstreamEvent] {
        let trimmed = json.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty, let data = trimmed.data(using: .utf8) else { return [] }

        guard let jsonObject = try? JSONSerialization.jsonObject(with: data) else { return [] }

        let rows: [[String: Any]]
        if let arr = jsonObject as? [[String: Any]] {
            rows = arr
        } else if let dict = jsonObject as? [String: Any] {
            rows = [dict]
        } else {
            return []
        }

        return rows.enumerated().map { (sequence, dict) in
            dictToEvent(dict: dict, sequence: sequence)
        }
    }

    private static func dictToEvent(dict: [String: Any], sequence: Int) -> ClickstreamEvent {
        let segmentation = (dict["segmentation"] as? [String: Any]) ?? [:]

        let eventId: String
        if let idDict = dict["_id"] as? [String: Any], let oid = idDict["$oid"] as? String, !oid.isEmpty {
            eventId = oid
        } else if let idStr = dict["_id"] as? String, !idStr.isEmpty {
            eventId = idStr
        } else {
            eventId = "row-\(sequence)"
        }

        let clientTimeValue = dict["client_time"] ?? dict["timestamp"]
        let clientTimeText: String?
        if let s = clientTimeValue as? String {
            clientTimeText = s
        } else if let n = clientTimeValue as? NSNumber {
            clientTimeText = "\(n.int64Value)"
        } else {
            clientTimeText = nil
        }

        let createdAtDate = (dict["created_at"] as? [String: Any])?["$date"] as? String
        let createdAt = clientTimeText ?? createdAtDate
        let timestamp = parseClientTimeToMillis(clientTimeValue)

        let customerId: String?
        if let rawCust = dict["customer_id"] {
            let str = "\(rawCust)".trimmingCharacters(in: .whitespacesAndNewlines)
            if str.isEmpty || ["null", "none", "nan"].contains(str.lowercased()) {
                customerId = nil
            } else if str.hasSuffix(".0") && str.dropLast(2).allSatisfy({ $0.isNumber }) {
                customerId = String(str.dropLast(2))
            } else {
                customerId = str
            }
        } else {
            customerId = nil
        }

        let platform = firstNonBlank([
            dict["platform"] as? String,
            segmentation["segment"] as? String,
            segmentation["segmentation.segment"] as? String,
            "iOS"
        ])

        let name = firstNonBlank([
            dict["segmentation_name"] as? String,
            segmentation["name"] as? String,
            segmentation["segmentation.name"] as? String,
            dict["key"] as? String,
            "unknown"
        ])

        let screenId = firstNonBlank([
            dict["screen_id"] as? String,
            segmentation["screen_id"] as? String,
            segmentation["segmentation.screen_id"] as? String
        ])

        let visit = firstNonBlankOrNil([
            (segmentation["visit"] != nil) ? "\(segmentation["visit"]!)" : nil,
            segmentation["segmentation.visit"] as? String
        ])

        return ClickstreamEvent(
            sourceIndex: sequence,
            eventId: eventId,
            deviceId: dict["device_id"] as? String ?? "",
            customerId: customerId,
            sessionId: dict["session_id"] as? String ?? "",
            createdAt: createdAt,
            timestamp: timestamp,
            key: dict["key"] as? String ?? "unknown",
            segment: platform,
            name: name,
            screenId: screenId,
            durationSeconds: 0,
            visit: visit
        )
    }

    private static func parseCsvRows(_ csv: String) -> [[String]] {
        var rows: [[String]] = []
        var row: [String] = []
        var field = ""
        var quoted = false
        let chars = Array(csv)
        var index = 0

        func finishField() {
            row.append(field)
            field = ""
        }

        func finishRow() {
            finishField()
            if !row.isEmpty { rows.append(row) }
            row = []
        }

        while index < chars.count {
            let char = chars[index]
            switch char {
            case "\"":
                if quoted && index + 1 < chars.count && chars[index + 1] == "\"" {
                    field.append("\"")
                    index += 1
                } else {
                    quoted = !quoted
                }
            case ",":
                if quoted {
                    field.append(char)
                } else {
                    finishField()
                }
            case "\n":
                if quoted {
                    field.append(char)
                } else {
                    finishRow()
                }
            case "\r":
                break
            default:
                field.append(char)
            }
            index += 1
        }
        if !field.isEmpty || !row.isEmpty {
            finishRow()
        }
        return rows
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

    public static func parseClientTimeToMillis(_ value: Any?) -> Int64 {
        guard let value = value else { return 0 }
        if let num = value as? NSNumber { return num.int64Value }
        if let s = value as? String {
            let trimmed = s.trimmingCharacters(in: .whitespacesAndNewlines)
            if let direct = Int64(trimmed) { return direct }
            if let date = isoWithMillis.date(from: trimmed) ?? isoStandard.date(from: trimmed) {
                return Int64(date.timeIntervalSince1970 * 1000.0)
            }
        }
        return 0
    }

    private static func firstNonBlank(_ values: [String?]) -> String {
        for v in values {
            if let str = v?.trimmingCharacters(in: .whitespacesAndNewlines), !str.isEmpty {
                return str
            }
        }
        return ""
    }

    private static func firstNonBlankOrNil(_ values: [String?]) -> String? {
        for v in values {
            if let str = v?.trimmingCharacters(in: .whitespacesAndNewlines), !str.isEmpty {
                return str
            }
        }
        return nil
    }
}
