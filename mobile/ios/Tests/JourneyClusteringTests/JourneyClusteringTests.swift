import XCTest
@testable import JourneyClustering

final class JourneyClusteringTests: XCTestCase {

    // MARK: - 1. Float16Helper & Math Tests

    func testFloat16HelperConversion() {
        // 1.0 in float16 bit pattern is 0x3C00
        let f1 = Float16Helper.halfToFloat(UInt16(0x3C00))
        XCTAssertEqual(f1, 1.0, accuracy: 1e-5)

        // 0.0 in float16 is 0x0000
        let f0 = Float16Helper.halfToFloat(UInt16(0x0000))
        XCTAssertEqual(f0, 0.0, accuracy: 1e-5)

        // -2.0 in float16 is 0xC000
        let fNeg2 = Float16Helper.halfToFloat(UInt16(0xC000))
        XCTAssertEqual(fNeg2, -2.0, accuracy: 1e-5)

        var vec: [Float] = [3.0, 4.0]
        Float16Helper.l2Normalize(&vec)
        XCTAssertEqual(vec[0], 0.6, accuracy: 1e-5)
        XCTAssertEqual(vec[1], 0.8, accuracy: 1e-5)
    }

    // MARK: - 2. SemanticTaxonomy Tests

    func testSemanticTaxonomyNormalization() {
        let path1 = "internet_fprotect_screen/management_device/management_device_screen/devices_details/device_information"
        let words1 = SemanticTaxonomy.normalizePath(path1)
        XCTAssertTrue(words1.contains("device"))
        XCTAssertTrue(words1.contains("info") || words1.contains("information") || words1.contains("detail"))

        let screenLabel = SemanticTaxonomy.classifyScreen(path1)
        XCTAssertEqual(screenLabel.businessFamily, "internet")
        XCTAssertEqual(screenLabel.operation, "view")
        XCTAssertTrue(screenLabel.isKnown)

        let actionLabel = SemanticTaxonomy.classifyAction(target: "btn_back", context: screenLabel)
        XCTAssertEqual(actionLabel.businessFamily, "internet")
        XCTAssertEqual(actionLabel.operation, "back")
        XCTAssertEqual(actionLabel.operationStage, "abort")
    }

    // MARK: - 3. Preprocessor Canonicalization & Cycle Collapse Tests

    func testPreprocessorRunAndCycleCollapse() {
        let prep = JourneyPreprocessor()

        func makeClick(token: String, timeMs: Int64, eventType: String = "view", name: String = "test") -> CanonicalClick {
            let raw = ClickstreamEvent(
                sourceIndex: 0,
                eventId: "e1",
                deviceId: "d1",
                sessionId: "s1",
                timestamp: timeMs,
                key: eventType,
                segment: "iOS",
                name: name
            )
            return CanonicalClick(
                raw: raw,
                eventType: eventType,
                segmentName: name,
                screenContext: name,
                eventToken: token,
                eventTimeMillis: timeMs,
                gapPrevSeconds: 1.0,
                gapNextSeconds: 1.0
            )
        }

        // Test consecutive run dedup: A A A B B C -> A B C
        let runs = [
            makeClick(token: "A", timeMs: 1000),
            makeClick(token: "A", timeMs: 2000),
            makeClick(token: "A", timeMs: 3000),
            makeClick(token: "B", timeMs: 4000),
            makeClick(token: "B", timeMs: 5000),
            makeClick(token: "C", timeMs: 6000)
        ]
        let keptRuns = prep.keepAfterRuns(runs)
        XCTAssertEqual(keptRuns.map { $0.eventToken }, ["A", "B", "C"])

        // Test cyclic repeats: A B A B A B -> A B
        let cycles = [
            makeClick(token: "A", timeMs: 1000),
            makeClick(token: "B", timeMs: 2000),
            makeClick(token: "A", timeMs: 3000),
            makeClick(token: "B", timeMs: 4000),
            makeClick(token: "A", timeMs: 5000),
            makeClick(token: "B", timeMs: 6000)
        ]
        let keptCycles = prep.keepAfterCycles(cycles)
        XCTAssertEqual(keptCycles.map { $0.eventToken }, ["A", "B"])
    }

    func testJourneySegmentation() {
        let prep = JourneyPreprocessor()
        prep.idleGapSeconds = 30.0

        let events = [
            ClickstreamEvent(sourceIndex: 0, eventId: "1", deviceId: "d1", sessionId: "s1", timestamp: 1000, key: "view", segment: "iOS", name: "HomeVC"),
            ClickstreamEvent(sourceIndex: 1, eventId: "2", deviceId: "d1", sessionId: "s1", timestamp: 5000, key: "view", segment: "iOS", name: "ServiceVC"),
            // Idle gap > 30s
            ClickstreamEvent(sourceIndex: 2, eventId: "3", deviceId: "d1", sessionId: "s1", timestamp: 45000, key: "view", segment: "iOS", name: "SupportVC"),
            ClickstreamEvent(sourceIndex: 3, eventId: "4", deviceId: "d1", sessionId: "s1", timestamp: 48000, key: "view", segment: "iOS", name: "HelpVC")
        ]

        let journeys = prep.prepareAll(events: events)
        XCTAssertEqual(journeys.count, 2)
        XCTAssertEqual(journeys[0].boundaryReason, "session_start")
        XCTAssertEqual(journeys[1].boundaryReason, "idle_gap")
    }

    // MARK: - 4. CSV & JSON Parsing Tests

    func testClickstreamRepositoryCsvAndJsonParsing() {
        let csvSample = """
        _id,device_id,customer_id,session_id,client_time,key,segment,segmentation_name,screen_id
        row1,dev1,10320319.0,sess1,1773851129000,[cly]_view,iOS,HomeVC,screen1
        row2,dev1,10320319.0,sess1,1773851135000,[cly]_action,iOS,btn_back,screen1
        """
        let csvEvents = ClickstreamRepository.parseCsvEvents(csvSample)
        XCTAssertEqual(csvEvents.count, 2)
        XCTAssertEqual(csvEvents[0].customerId, "10320319")
        XCTAssertEqual(csvEvents[0].sessionId, "sess1")
        XCTAssertEqual(csvEvents[0].key, "[cly]_view")
        XCTAssertEqual(csvEvents[0].timestamp, 1773851129000)

        let jsonSample = """
        [
            {
                "_id": "60a1b2c3d4",
                "device_id": "dev2",
                "customer_id": "998877",
                "session_id": "sess2",
                "timestamp": 1773851200000,
                "key": "action",
                "segmentation": {
                    "segment": "iOS",
                    "name": "btn_pay",
                    "screen_id": "PaymentVC"
                }
            }
        ]
        """
        let jsonEvents = ClickstreamRepository.parseJsonEvents(jsonSample)
        XCTAssertEqual(jsonEvents.count, 1)
        XCTAssertEqual(jsonEvents[0].customerId, "998877")
        XCTAssertEqual(jsonEvents[0].name, "btn_pay")
        XCTAssertEqual(jsonEvents[0].screenId, "PaymentVC")
    }

    // MARK: - 5. End-to-End JourneyModel Integration Test with Assets

    func testGuestLoginGateClusterRemainsHome() throws {
        let thisFile = URL(fileURLWithPath: #file)
        let packageRoot = thisFile
            .deletingLastPathComponent()
            .deletingLastPathComponent()
            .deletingLastPathComponent()
        let mappingURL = packageRoot
            .appendingPathComponent("resource")
            .appendingPathComponent("cluster_mapping.json")
        let mapping = try ClusterMapping(
            json: String(contentsOf: mappingURL, encoding: .utf8)
        )

        let guestLogin = try XCTUnwrap(mapping.lookup(clusterId: 531))
        XCTAssertEqual(guestLogin.clusterName, "Home | Trang chủ")
        XCTAssertEqual(guestLogin.businessFamily, "Home")
        XCTAssertEqual(guestLogin.businessSubmodule, "Trang chủ")
    }

    func testJourneyModelWithWorkspaceAssets() throws {
        // Locate resource directory relative to this test file (portable across any machine/environment)
        let thisFile = URL(fileURLWithPath: #file)
        let packageRoot = thisFile
            .deletingLastPathComponent() // Tests/JourneyClusteringTests
            .deletingLastPathComponent() // Tests
            .deletingLastPathComponent() // mobile/ios
        let resourceDir = packageRoot.appendingPathComponent("resource")

        guard FileManager.default.fileExists(atPath: resourceDir.appendingPathComponent("manifest.json").path) else {
            XCTFail("Assets directory not found at relative path: \(resourceDir.path)")
            return
        }

        let model = try JourneyModel(directoryURL: resourceDir)
        XCTAssertGreaterThan(model.centroidAssigner.clusterIds.count, 0)
        XCTAssertEqual(model.vectorizer.embeddingDim, 48)

        // Load test data CSV from resource directory
        let csvPath = resourceDir.appendingPathComponent("test_data.csv").path
        let testCsvUrl = FileManager.default.fileExists(atPath: csvPath)
            ? resourceDir.appendingPathComponent("test_data.csv")
            : resourceDir.appendingPathComponent("test_data_android.csv")

        let csvData = try String(contentsOf: testCsvUrl, encoding: .utf8)
        let sessions = ClickstreamRepository.loadSessions(content: csvData, isCsv: true)
        XCTAssertFalse(sessions.isEmpty)

        // Pick the session with largest events count
        let session = sessions.first!
        let result = model.analyzeEvents(session.events)
        XCTAssertFalse(result.predictions.isEmpty)

        let firstPred = result.predictions.first!
        XCTAssertNotNil(firstPred.cluster)
        XCTAssertNotNil(firstPred.clusterName)
        XCTAssertNotNil(firstPred.distanceToCentroid)
        print("Scored Journey: \(firstPred.journeyId), Cluster: \(firstPred.cluster ?? -1), Name: \(firstPred.clusterName ?? ""), Distance: \(firstPred.distanceToCentroid ?? 0.0), Logprob: \(firstPred.markovLogprob ?? 0.0)")

        print("analyzeEvents predictions count: \(result.predictions.count)")
        for (idx, p) in result.predictions.enumerated() {
            print("  [Batch \(idx)] \(p.journeyId) reason: \(p.boundaryReason) events: \(p.eventsSeen)")
        }

        // Stream simulation with ClickstreamProcessor
        let processor = ClickstreamProcessor(model: model, route: .journeyComplete)
        processor.start()

        var emittedPredictions: [MobilePrediction] = []
        for (i, event) in session.events.enumerated() {
            let update = processor.accept(event: event)
            if !update.finalized.isEmpty {
                for p in update.finalized {
                    print("  [Stream event \(i)] finalized \(p.journeyId) reason: \(p.boundaryReason) events: \(p.eventsSeen)")
                }
                emittedPredictions.append(contentsOf: update.finalized)
            }
        }
        if let finishUpdate = processor.finish(sessionId: session.id) {
            for p in finishUpdate.finalized {
                print("  [Stream finish] finalized \(p.journeyId) reason: \(p.boundaryReason) events: \(p.eventsSeen)")
            }
            emittedPredictions.append(contentsOf: finishUpdate.finalized)
        }

        XCTAssertFalse(emittedPredictions.isEmpty)
        XCTAssertEqual(emittedPredictions.count, result.predictions.count)
    }

    func testRelativePathInitialization() throws {
        // Test initializing with relative URL "resource" from current working directory
        let relativeURL = URL(fileURLWithPath: "resource")
        if FileManager.default.fileExists(atPath: relativeURL.appendingPathComponent("manifest.json").path) {
            let model = try JourneyModel(directoryURL: relativeURL)
            XCTAssertEqual(model.vectorizer.embeddingDim, 48)
            XCTAssertGreaterThan(model.centroidAssigner.clusterIds.count, 0)
        }
    }
}
