# Swift / iOS Journey Clustering SDK

A production-grade, zero-external-dependency Swift port of the Hi FPT unsupervised clickstream journey clustering, anomaly detection, and streaming inference pipeline.

---

## 🏗️ Architecture & Component Separation

The iOS pipeline is partitioned into distinct components matching the Android (`vn.hifpt.clickstream`) and Python architectures:

```
Sources/JourneyClustering/
├── Models/
│   └── JourneyModels.swift          # Event, Click, Journey, Prediction & Output models
├── Preprocessing/
│   ├── SemanticTaxonomy.swift       # Business taxonomy classification (family, module, object, operation)
│   └── JourneyPreprocessor.swift    # Canonization, segmentation, run/cycle deduplication, numeric metrics
├── Vectorization/
│   ├── Float16Helper.swift          # IEEE 754 half-precision decoding & SIMD/Accelerate L2 normalization
│   └── JourneyVectorizer.swift      # Multi-channel SVD*IDF + scaled numeric + global PCA transformer
├── Scoring/
│   ├── CentroidAssigner.swift       # Euclidean cluster centroid assignment
│   ├── MarkovBank.swift             # Markov companion model (logprob scoring & next action prediction)
│   └── JourneyScorer.swift          # Distance scoring, anomaly detection, and friction diagnostics
├── PostProcessing/
│   ├── ClusterMapping.swift         # Business archetype definitions & cluster metadata
│   └── JourneyPostProcessor.swift   # Archetype mapping, naming fallback, confidence & prediction formatting
├── Streaming/
│   ├── JourneyStream.swift          # Thread-safe multi-session buffer & provisional scorer
│   └── ClickstreamProcessor.swift   # High-level facade for FIXED_DURATION and JOURNEY_COMPLETE routes
├── Storage/
│   └── ClickstreamRepository.swift  # Streaming CSV & JSON clickstream parser
└── Engine/
    └── JourneyModel.swift           # Unified model coordinator & bundle loader
```

---

## 📦 Core Components & Responsibilities

### 1. `JourneyDataModels` (`JourneyModels.swift`)
- **`ClickstreamEvent`**: Raw event parsed from SDK JSON or CSV logs (`sessionId`, `deviceId`, `customerId`, `timestamp`, `key`, `name`, `screenId`, etc.).
- **`CanonicalClick`**: Enriched event with canonical path masking (`{id}`, `{uuid}`, `{code}`), resolved screen context, multi-level hierarchical tokens (`tokenL1`, `tokenL2`, `tokenL3`, `operationToken`), and gap times.
- **`PreparedJourney`**: Fully cleaned journey ready for vectorization (`tokens`, multi-channel tokens, `numeric` metrics: `backRate`, `revisitRatio`, `spanSeconds`, `nLoopRemoved`, `nDedupRemoved`).
- **`MobilePrediction`**: Prediction result containing `cluster`, `clusterName`, `businessFamily`, `distanceToCentroid`, `markovLogprob`, `geometricAnomaly`, `generativeAnomaly`, `frictionFlags`, `nextAction`, and `medoidPath`.

### 2. `JourneyPreprocessor` (`JourneyPreprocessor.swift`)
- **Path Normalization**: Canonicalizes screen and action names, strips transient query parameters, and masks dynamic IDs/UUIDs/hex codes.
- **Semantic Classification**: Maps events into coarse and fine business categories (`internet`, `payment`, `account`, `support`, `loyalty`, `shop`, etc.) using `SemanticTaxonomy`.
- **Segmentation Boundaries**: Detects boundaries by `session_start`, `idle_gap` (> 90s), `length_cap` (>= 80 events), `root_return` (>= 6 events), and `auth_change`.
- **Sequence Collapse**:
  - `keepAfterRuns`: Collapses consecutive duplicate navigation fires (`A A A -> A`).
  - `keepAfterCycles`: Collapses periodic screen thrashing (`A B A B A B -> A B`), preserving cycle count as friction signal `nLoopRemoved`.

### 3. `JourneyVectorizer` (`JourneyVectorizer.swift`)
- Decodes IEEE 754 half-precision binary projection weights (`projection_weights.float16.bin`).
- Extracts 1-grams and 2-grams (with `<bos>` / `<eos>` bounds) across 4 channels:
  - `primary`: Exact screen/action tokens
  - `coarse`: L2 business modules
  - `intent`: L3 object + operation
  - `operation`: Stage + action
- Computes sublinear TF `1.0 + ln(count)` and projects through channel SVD weight matrices.
- Standardizes numerical features (with `log1p` transforms) and applies global PCA projection with L2 normalization.

### 4. `JourneyScorer` (`JourneyScorer.swift`)
- **Centroid Distance**: Calculates Euclidean distance against cluster centroids in 48-dimensional space.
- **Markov Companion Model**: Evaluates sequence transition likelihood $P(\text{journey} \mid \text{cluster})$ with Laplace smoothing.
- **Anomaly Detection**:
  - `geometricAnomaly`: $d > d_{95}$ (0.70734)
  - `generativeAnomaly`: $\text{logprob} < \text{markov}_{05}$ (-4.49426)
  - `severeAnomaly`: $\text{geometric} \land \text{logprob} < \text{markov}_{01}$ (-5.51838)
- **Friction Diagnostics**: Flags `excessive_back`, `navigation_loop`, `screen_thrash`, `slow_journey`, `unknown_archetype`, and `improbable_transitions`.

### 5. `JourneyPostProcessor` (`JourneyPostProcessor.swift`)
- Maps assigned `clusterId` to archetype metadata (`clusterName`, `businessFamily`, `businessSubmodule`, `businessDetail`, `namingConfidence`, `medoidPath`).
- Maps anomalous journeys to noise archetype (`-1`: "Chưa phân loại | Journey hỗn hợp/nhiễu").
- Resolves lifecycle state (`collecting`, `provisional`, `final`).

---

## 🚀 Quick Start & Integration

### A. Load Model in iOS App or CLI

All iOS model asset files are located in `mobile/ios/resource/`:
- `config.json`
- `vocabularies.json`
- `vectorizer_metadata.json`
- `projection_weights.float16.bin`
- `manifest.json`
- `centroids_matrix.bin`
- `markov.json`
- `cluster_mapping.json`
- `thresholds.json`

#### Option 1: Using Relative or Directory Path
```swift
import JourneyClustering

// Load model from relative folder path "resource" or custom URL
let model = try JourneyModel(directoryURL: URL(fileURLWithPath: "resource"))
```

#### Option 2: Using iOS App Bundle (Xcode)
Add the `resource` folder (or individual files) into your Xcode target:
```swift
import JourneyClustering

// Load model directly from Bundle.main
let model = try JourneyModel(bundle: .main, subfolder: "resource")
// or if files are placed directly in main bundle root:
// let model = try JourneyModel(bundle: .main)
```

### B. Batch Journey Analysis

```swift
// Ingest raw JSON or CSV events
let events: [ClickstreamEvent] = ClickstreamRepository.parseJsonEvents(rawJsonString)

// Run end-to-end analysis
let result: MobileAnalysisResult = model.analyzeEvents(events)

for prediction in result.predictions {
    print("Journey: \(prediction.journeyId)")
    print("Cluster: \(prediction.cluster ?? -1) - \(prediction.clusterName ?? "Unknown")")
    print("Distance: \(prediction.distanceToCentroid ?? 0.0)")
    print("Markov Logprob: \(prediction.markovLogprob ?? 0.0)")
    print("Friction Flags: \(prediction.frictionFlags)")
    print("Predicted Next Action: \(prediction.nextAction ?? "None")")
}
```

### C. Live Real-Time Stream Processing

```swift
// Choose route: .journeyComplete (instant clustering upon boundary) or .fixedDuration (30s windowing)
let processor = ClickstreamProcessor(model: model, route: .journeyComplete)
processor.start()

// Ingest live events as they occur in your iOS analytics tracker
func onAnalyticsEvent(name: String, screenId: String, key: String) {
    let event = ClickstreamEvent(
        sourceIndex: nextIndex(),
        eventId: UUID().uuidString,
        deviceId: UIDevice.current.identifierForVendor?.uuidString ?? "",
        customerId: currentUserId,
        sessionId: currentSessionId,
        timestamp: Int64(Date().timeIntervalSince1970 * 1000.0),
        key: key,
        segment: "iOS",
        name: name,
        screenId: screenId
    )

    let update = processor.accept(event: event)
    for finalized in update.finalized {
        print("Finalized Journey: \(finalized.clusterName ?? "")")
        // Post notification or upload to backend API
    }
}

// When session ends
if let finishUpdate = processor.finish(sessionId: currentSessionId) {
    for finalized in finishUpdate.finalized {
        print("Session End Journey: \(finalized.clusterName ?? "")")
    }
}
```

---

## 🧪 Testing

Run the test suite from the terminal:

```bash
cd mobile/ios
swift test
```
