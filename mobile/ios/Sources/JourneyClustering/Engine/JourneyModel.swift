import Foundation

/// Unified Journey Clustering Model Engine for Swift/iOS.
/// Coordinates preprocessing, vectorization, scoring, and post-processing.
public final class JourneyModel: @unchecked Sendable {
    public let preprocessor: JourneyPreprocessor
    public let vectorizer: JourneyVectorizer
    public let centroidAssigner: CentroidAssigner
    public let markov: MarkovBank
    public let scorer: JourneyScorer
    public let clusterMapping: ClusterMapping
    public let postProcessor: JourneyPostProcessor
    public let thresholds: ScoreThresholds

    public init(
        configJson: String,
        vocabJson: String,
        metaJson: String,
        weightsData: Data,
        manifestJson: String,
        centroidsData: Data,
        markovJson: String,
        mappingJson: String,
        thresholdsJson: String
    ) throws {
        self.preprocessor = JourneyPreprocessor()
        self.preprocessor.configure(configJson: configJson)

        self.vectorizer = try JourneyVectorizer.load(
            vocabJson: vocabJson,
            metaJson: metaJson,
            weightsData: weightsData
        )

        self.centroidAssigner = try CentroidAssigner.load(
            manifestJson: manifestJson,
            binaryCentroids: centroidsData
        )

        self.markov = try MarkovBank(markovJson: markovJson)
        self.clusterMapping = try ClusterMapping(json: mappingJson)

        if let threshData = thresholdsJson.data(using: .utf8),
           let tObj = try? JSONSerialization.jsonObject(with: threshData) as? [String: Any] {
            self.thresholds = ScoreThresholds(
                distanceP95: (tObj["distance_p95"] as? NSNumber)?.doubleValue ?? 0.70734,
                markovP05: (tObj["markov_p05"] as? NSNumber)?.doubleValue ?? -4.49426,
                markovP01: (tObj["markov_p01"] as? NSNumber)?.doubleValue ?? -5.51838,
                backRateP90: (tObj["back_rate_p90"] as? NSNumber)?.doubleValue ?? 0.20,
                loopsP90: (tObj["loops_p90"] as? NSNumber)?.doubleValue ?? 0.0,
                revisitP90: (tObj["revisit_p90"] as? NSNumber)?.doubleValue ?? 0.2609,
                spanP95: (tObj["span_p95"] as? NSNumber)?.doubleValue ?? 138.0
            )
        } else {
            self.thresholds = ScoreThresholds()
        }

        self.scorer = JourneyScorer(
            vectorizer: self.vectorizer,
            centroidAssigner: self.centroidAssigner,
            markov: self.markov,
            thresholds: self.thresholds
        )

        self.postProcessor = JourneyPostProcessor(
            clusterMapping: self.clusterMapping,
            thresholds: self.thresholds,
            minJourneyLength: self.preprocessor.minJourneyLength
        )
    }

    /// Convenience initializer from Bundle file URLs or directory URL (supports relative and absolute paths).
    public convenience init(directoryURL: URL) throws {
        let stdURL = directoryURL.standardizedFileURL
        let configJson = try String(contentsOf: stdURL.appendingPathComponent("config.json"), encoding: .utf8)
        let vocabJson = try String(contentsOf: stdURL.appendingPathComponent("vocabularies.json"), encoding: .utf8)
        let metaJson = try String(contentsOf: stdURL.appendingPathComponent("vectorizer_metadata.json"), encoding: .utf8)
        let weightsData = try Data(contentsOf: stdURL.appendingPathComponent("projection_weights.float16.bin"))
        let manifestJson = try String(contentsOf: stdURL.appendingPathComponent("manifest.json"), encoding: .utf8)
        let centroidsData = try Data(contentsOf: stdURL.appendingPathComponent("centroids_matrix.bin"))
        let markovJson = try String(contentsOf: stdURL.appendingPathComponent("markov.json"), encoding: .utf8)
        let mappingJson = try String(contentsOf: stdURL.appendingPathComponent("cluster_mapping.json"), encoding: .utf8)
        let thresholdsJson = try String(contentsOf: stdURL.appendingPathComponent("thresholds.json"), encoding: .utf8)

        try self.init(
            configJson: configJson,
            vocabJson: vocabJson,
            metaJson: metaJson,
            weightsData: weightsData,
            manifestJson: manifestJson,
            centroidsData: centroidsData,
            markovJson: markovJson,
            mappingJson: mappingJson,
            thresholdsJson: thresholdsJson
        )
    }

    /// Convenience initializer from an iOS Bundle (e.g. Bundle.main or Bundle.module).
    public convenience init(bundle: Bundle, subfolder: String? = nil) throws {
        func fileURL(name: String, ext: String) throws -> URL {
            if let subfolder = subfolder,
               let url = bundle.url(forResource: name, withExtension: ext, subdirectory: subfolder) {
                return url
            }
            if let url = bundle.url(forResource: name, withExtension: ext) {
                return url
            }
            if let resourceSub = bundle.url(forResource: "\(name).\(ext)", withExtension: nil) {
                return resourceSub
            }
            throw NSError(
                domain: "JourneyModel",
                code: 404,
                userInfo: [NSLocalizedDescriptionKey: "Model resource \(name).\(ext) not found in bundle \(bundle)"]
            )
        }

        let configUrl = try fileURL(name: "config", ext: "json")
        let vocabUrl = try fileURL(name: "vocabularies", ext: "json")
        let metaUrl = try fileURL(name: "vectorizer_metadata", ext: "json")
        let weightsUrl = try fileURL(name: "projection_weights.float16", ext: "bin")
        let manifestUrl = try fileURL(name: "manifest", ext: "json")
        let centroidsUrl = try fileURL(name: "centroids_matrix", ext: "bin")
        let markovUrl = try fileURL(name: "markov", ext: "json")
        let mappingUrl = try fileURL(name: "cluster_mapping", ext: "json")
        let thresholdsUrl = try fileURL(name: "thresholds", ext: "json")

        try self.init(
            configJson: try String(contentsOf: configUrl, encoding: .utf8),
            vocabJson: try String(contentsOf: vocabUrl, encoding: .utf8),
            metaJson: try String(contentsOf: metaUrl, encoding: .utf8),
            weightsData: try Data(contentsOf: weightsUrl),
            manifestJson: try String(contentsOf: manifestUrl, encoding: .utf8),
            centroidsData: try Data(contentsOf: centroidsUrl),
            markovJson: try String(contentsOf: markovUrl, encoding: .utf8),
            mappingJson: try String(contentsOf: mappingUrl, encoding: .utf8),
            thresholdsJson: try String(contentsOf: thresholdsUrl, encoding: .utf8)
        )
    }

    /// End-to-end analysis of raw events into a finalized MobileAnalysisResult.
    public func analyzeEvents(_ events: [ClickstreamEvent]) -> MobileAnalysisResult {
        let preparedJourneys = preprocessor.prepareAll(events: events)
        let predictions = preparedJourneys.map { scorePrepared($0, provisional: false) }
        return MobileAnalysisResult(predictions: predictions)
    }

    /// Score a single prepared journey.
    public func scorePrepared(_ prepared: PreparedJourney, provisional: Bool) -> MobilePrediction {
        if prepared.nEventsFinal < preprocessor.minJourneyLength {
            return postProcessor.postProcessCollecting(prepared: prepared, provisional: provisional)
        }
        let scoring = scorer.score(prepared: prepared)
        return postProcessor.postProcess(prepared: prepared, scoring: scoring, provisional: provisional)
    }

    /// Score a sequence of events for a journey ID.
    public func scoreEvents(
        journeyId: String,
        events: [ClickstreamEvent],
        provisional: Bool,
        boundaryReason: String = ""
    ) -> MobilePrediction? {
        guard let prepared = preprocessor.prepareSingle(journeyId: journeyId, events: events, boundaryReason: boundaryReason) else {
            return nil
        }
        return scorePrepared(prepared, provisional: provisional)
    }
}
