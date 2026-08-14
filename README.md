# Unsupervised Journey Clustering

This repository converts raw mobile clickstream events into reusable, unsupervised
journey archetypes. It supports Android and iOS, detects unusual journeys, explains
friction signals, and can score new production events without retraining.

The main training entry point is `scripts/run_journey_pipeline.py`.

## What the system does

```mermaid
flowchart LR
    A[Raw production CSVs] --> B[Canonicalise events]
    B --> C[Semantic enrichment]
    C --> D[Multi-resolution tokens]
    D --> E[Session to journey segmentation]
    E --> F[Sequence cleanup and numeric features]
    F --> G[TF-IDF n-grams + SVD]
    G --> H[HDBSCAN archetypes]
    H --> I[Markov anomaly scoring]
    I --> J[Scorer and reports]
```

The pipeline is unsupervised: it does not require a labelled journey or business-
intent column. HDBSCAN discovers dense behavioural groups and leaves low-density
journeys as noise (`cluster = -1`) instead of forcing them into a known cluster.

## Technical approach

### 1. Canonicalisation

The raw event contract is normalised into a stable event-level schema:

- `View` events use `segment_name` as the screen.
- `Action` events use `screen_id` or the most recent view as screen context, and
  retain the action path as the target.
- URLs keep semantic query parameters but remove instance identifiers, timestamps,
  UUIDs, and ID-like path segments.
- Platform is retained explicitly in the canonical `platform` column, and Android
  and iOS are trained/scored independently.
- Every event receives an exact token such as `view@HomeVC` or
  `action@HomeVC#profile/open`.

This removes telemetry noise without merging genuinely different screens.

### 2. Semantic enrichment and tokens

The taxonomy derives business meaning from screen and action paths using normalized
words, aliases, and phrase matching. Each event receives semantic fields including
business family, module, object, operation, and operation stage.

The same event is represented at several resolutions:

| Level | Example | Use |
|---|---|---|
| `EXACT` | `action@HomeVC#profile/open` | Faithful path; platform remains in `platform` |
| `L3` | `account/profile/user/open` | Intent-like family/module/object/operation |
| `L2` | `account/profile` | Coarser module behaviour |
| `L1` | `account` | Broad business family |
| operation | `navigate:open` | Browse/configure/commit-style action shape |

Rare tokens are folded to a configured backoff level, then to `<rare>`, based on
document frequency across journeys rather than repeated events inside one journey.

### 3. Journey segmentation

Raw `session_id` values can contain multiple unrelated user goals, so the pipeline
creates shorter journeys. A boundary is created by:

- a new session;
- an idle gap greater than `90` seconds by default;
- a return to a root or hub screen after at least six events;
- a login/logout action change; or
- the maximum journey length (`80` events by default).

Optional branching-entropy boundaries can be enabled with `--entropy`. Journeys
shorter than `--min-journey-length` are retained in the full export but excluded
from clustering.

### 4. Sequence cleanup and behavioural features

The post-processing stage:

- collapses consecutive duplicate events;
- collapses repeated cycles of period 2–4 while retaining the number removed as a
  loop feature;
- optionally removes OS chrome and boot screens from the modelling sequence; and
- computes length, action ratio, back rate, revisit ratio, loop count, span, and gap
  statistics.

Removed information is not silently discarded: it remains available in the numeric
feature block and journey-level output columns.

### 5. Journey representation

The active route uses `JourneyVectorizer`:

1. Treat each cleaned journey as a document with `<bos>` and `<eos>` markers.
2. Build contiguous TF-IDF n-grams, defaulting to 1–3 grams.
3. Reduce the sparse sequence block with Truncated SVD, defaulting to 64 dimensions.
4. Independently encode the optional `coarse`, `intent`, and `operation` semantic
   channels, each with its own TF-IDF/SVD block.
5. Append a standardized numeric behaviour block with default weight `0.35`.
6. L2-normalize each block before applying its configured weight and concatenating.

This lets journeys with different screen names remain close when their business
intent and operation shape are similar.

The older PrefixSpan/pattern-vectorization route remains documented in
`docs/4_Clustering_Two_Routes.md`, but it is not invoked by the current main entry
point.

### 6. Clustering and anomaly detection

- **HDBSCAN** discovers clusters without requiring a fixed number of clusters.
  `min_cluster_size` controls the minimum archetype size; `min_samples` controls
  core-point strictness. The `eom` or `leaf` selection method is configurable.
- **KMeans** is run over a small `k` grid as a baseline and sanity check. It is not
  the production assignment model.
- **MarkovBank** fits a smoothed first-order transition model per cluster plus a
  global fallback chain. It provides length-normalized transition log-probability
  and next-action prediction.
- Training thresholds are calibrated from the fitted journeys. New journeys can be
  flagged as geometrically anomalous, generatively anomalous, or severe when both
  signals fire.

The main friction flags are `excessive_back`, `navigation_loop`, `screen_thrash`,
`slow_journey`, `unknown_archetype`, and `improbable_transitions`.

## Repository layout

| Path | Purpose |
|---|---|
| `src/` | Canonicalisation, taxonomy, semantics, tokenisation, segmentation, features, clustering, scoring |
| `scripts/run_journey_pipeline.py` | End-to-end training and holdout verification |
| `scripts/score_new_events.py` | Inference on new raw events |
| `scripts/postprocess_cluster_runs.py` | Optional strict C / lenient B hierarchical post-processing |
| `scripts/name_hierarchical_clusters.py` | Generate names for hierarchical clusters |
| `scripts/export_scorer_to_onnx.py` | Export a compatible single-channel scorer for mobile/ONNX |
| `dashboard/` | Streamlit dashboard and cache preparation |
| `tests/` | Unit tests for canonicalisation, segmentation, tokens, scoring, and post-processing |
| `docs/` | Detailed technical and validation notes |

## Configuration and prerequisites

### Python environment

Use Python 3.10+ and install the main dependencies:

```powershell
py -3 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Install the optional ONNX dependencies only when exporting or validating a mobile
bundle:

```powershell
python -m pip install -r requirements-onnx.txt
```

The raw data is intentionally excluded by `.gitignore`; provide the input files
locally before running the pipeline.

### Raw input schema

The production CSVs must contain:

```text
session_id, platform, key, segmentation_name, client_time
```

Recommended optional columns are `device_id`, `customer_id`, `screen_id`, and `_id`.
The adapter also accepts these aliases: `timestamp` for `client_time`,
`segmentation.segment` for `platform`, `segmentation.name` for
`segmentation_name`, and `segmentation.screen_id` for `screen_id`.

For training, `--input` must point to a directory; the directory is read recursively
and all CSV files are combined. For inference, `scripts/score_new_events.py` accepts
a single CSV through `--input`.

### Main training options

| Option | Default | Meaning |
|---|---:|---|
| `--input` | `data/train_data/raw_data_production/data_raw_sample` | Folder of raw event CSVs |
| `--output-root` | `output/journey_runs` | Root for prepared data and run folders |
| `--test-size` | `0.2` | Share of latest complete sessions held out |
| `--level` | `EXACT` | Primary token resolution: `EXACT`, `L3`, `L2`, or `L1` |
| `--backoff-level` | `L2` | Fallback resolution for rare primary tokens |
| `--channel-weights` | `coarse=0.45,intent=0.30,operation=0.20` | Semantic feature-channel weights; empty string disables them |
| `--idle-gap` | `90` | Seconds that split a journey |
| `--min-journey-length` | `4` | Minimum cleaned events used for clustering |
| `--token-min-df` | `3` | Minimum journey document frequency before rare folding |
| `--ngram-min`, `--ngram-max` | `1`, `3` | Contiguous n-gram range |
| `--feature-min-df` | `3` | Minimum document frequency for TF-IDF features |
| `--max-features` | `20000` | Maximum TF-IDF vocabulary size per channel |
| `--svd-dim` | `64` | SVD dimensions per sequence block |
| `--numeric-weight` | `0.35` | Weight of numeric behaviour features |
| `--min-cluster-size` | `100` | HDBSCAN minimum cluster size |
| `--min-samples` | `5` | HDBSCAN core-point strictness |
| `--cluster-selection-method` | `eom` | HDBSCAN selection: `eom` or `leaf` |
| `--entropy` | off | Enable branching-entropy refinement boundaries |
| `--keep-chrome` | off | Keep OS navigation/container screens in the sequence |
| `--drop-boot` | off | Remove boot/splash screens from the sequence |
| `--preprocess` | off | Rebuild the canonical train/test cache |

The run folder name is generated from these settings, for example:
`EXACT_ch-c45i30o20_ng1-3_svd64_...`. This makes configurations reproducible and
prevents different experiments from overwriting one another.

## Run the training pipeline

For the data layout currently present in this repository, use the parent folder
containing the six raw CSV files:

```powershell
python scripts/run_journey_pipeline.py `
  --input data/train_data `
  --output-root output/journey_runs `
  --test-size 0.2
```

Force preprocessing after changing the input data or segmentation settings:

```powershell
python scripts/run_journey_pipeline.py `
  --input data/train_data `
  --output-root output/journey_runs `
  --preprocess
```

Example tuning run:

```powershell
python scripts/run_journey_pipeline.py `
  --input data/train_data `
  --min-cluster-size 50 `
  --min-samples 3 `
  --ngram-min 1 `
  --ngram-max 4 `
  --svd-dim 128 `
  --level L2
```

Training splits Android and iOS independently. The newest complete sessions form
the holdout set, so no session is split between train and test and no event-level
leakage is introduced. Prepared canonical data is cached under
`<output-root>/_prepared/` and reused when only clustering/feature settings change.

### Important checkout note

In this checkout, modules are stored directly under `src/`, while scripts and tests
import them as `Rule_based` (for example, `from Rule_based.config import ...`). The
repository currently does not contain a `src/Rule_based/` package or packaging
metadata that provides this alias. A clean run therefore fails at import time until
the package layout/import alias is restored or the imports are made consistent.

Also, the script's built-in input default points to
`data/train_data/raw_data_production/data_raw_sample`, which is not present in this
checkout; pass `--input data/train_data` or your own CSV/folder explicitly.

## What the training run outputs

The generated directory contains run provenance plus one set of artifacts for each
platform (`android_...` and `ios_...`).

### Run-level files

| File | Contents |
|---|---|
| `experiment_config.json` | Complete run manifest, input, test size, and serialized configuration |
| `preprocessing_report.csv` | Input-file validation and dropped/invalid-row counts |
| `split_report.csv` | Chronological train/holdout session split and leakage checks |
| `_prepared/` | Cached canonical Android/iOS train and holdout events |

### Platform-level files

| File | Contents |
|---|---|
| `<platform>_events_canonical.csv` | Event-level canonical, semantic, token, journey, and timing fields |
| `<platform>_token_dictionary.csv` | Token frequencies and provenance for the selected level |
| `<platform>_journeys.csv` | Clean journeys used for clustering, one row per journey |
| `<platform>_journeys_all_including_short.csv` | All segmented journeys, including those below the minimum length |
| `<platform>_sequences.jsonl` | Journey IDs, primary token sequences, and aligned semantic channels |
| `<platform>_cluster_catalog.json` | One row per HDBSCAN cluster/noise group, with size, share, medoid, and behaviour statistics |
| `<platform>_cluster_ngrams.csv` | Representative n-grams for interpreting each cluster |
| `<platform>_journey_scorer.pkl` | Fitted vectorizer, centroids, Markov models, and anomaly thresholds |
| `<platform>_scored_holdout.csv` | Holdout journeys scored against the fitted model, when the holdout is large enough |
| `<platform>_run_config.json` | Configuration, feature dimensions, HDBSCAN metrics, and noise share |
| `<platform>_RUN_REPORT.md` | Human-readable report of pipeline-stage tables |
| `<platform>_report_*.csv` | Canonisation, segmentation, cleanup, rare-folding, KMeans, and cluster reports |

The most important columns in `<platform>_journeys.csv` and scored outputs include:

`journey_id`, `session_id`, `cluster`, `sequence`, `n_events_final`,
`action_ratio`, `back_rate`, `revisit_ratio`, `n_loop_removed`, `span_seconds`,
`markov_logprob`, `geometric_anomaly`, `generative_anomaly`, `severe_anomaly`,
`friction_flags`, `next_action`, and `next_action_share`.

## Score new production events

### Single fitted run

```powershell
python scripts/score_new_events.py `
  --platform android `
  --input data/new_events.csv `
  --single-run output/journey_runs/<run-folder> `
  --output output/test/android_scored.csv
```

The scorer re-runs the same canonicalisation, tokenisation, segmentation, cleanup,
and vectorization pipeline. It does not refit TF-IDF, SVD, clusters, or thresholds.

Smoke-test the latest day of a training CSV:

```powershell
python scripts/score_new_events.py `
  --platform ios `
  --input data/train_data/ios_t5_1k.csv `
  --single-run output/journey_runs/<run-folder> `
  --holdout-days 1 `
  --predict-next
```

### Optional hierarchical C/B inference

For higher coverage, fit two runs with identical preprocessing but different
HDBSCAN strictness. Use the strict run as primary C and the lenient run as fallback B:

```powershell
python scripts/postprocess_cluster_runs.py `
  --b-run output/journey_runs/<lenient-run> `
  --c-run output/journey_runs/<strict-run> `
  --name C_primary_B_secondary
```

Then score with:

```powershell
python scripts/score_new_events.py `
  --platform android `
  --input data/new_events.csv `
  --postprocess-run output/journey_runs/C_primary_B_secondary `
  --output output/test/android_hierarchical_scored.csv
```

Assignments are namespaced as `C:<id>`, `B:<id>`, or `UNKNOWN`. A B fallback is
accepted only when it passes cluster distance, Markov-likelihood, and nearest-vs-
second-nearest margin gates. Original C labels are preserved.

## Dashboard

The Streamlit dashboard reads a selected fitted run and compact cache files:

```powershell
python dashboard/prepare_dashboard_data.py
python -m streamlit run dashboard/app.py
```

The precompute step produces `output/dashboard_cache/eda.json`, `train_run.json`,
`inference_android.parquet`, `inference_ios.parquet`, and `showcase.json`. Rebuild
the cache after every new training run. See `dashboard/README.md` for the dashboard-
specific run and theme details.

## Validation and interpretation

Useful checks after training:

1. Read `<platform>_run_config.json` for `n_clusters`, `noise_share`, silhouette,
   Davies–Bouldin, and Calinski–Harabasz metrics.
2. Inspect `<platform>_cluster_catalog.json` and `<platform>_cluster_ngrams.csv`;
   cluster size alone is not evidence of a meaningful archetype.
3. Compare holdout `cluster != -1` coverage and anomaly rates across platforms.
4. Review `friction_flags` and representative `sequence` paths for product meaning.
5. Retrain after major navigation/taxonomy changes or when known-archetype coverage
   drops materially on fresh data.

Detailed design rationale and validation notes are in `docs/`, especially:

- `docs/0_Solution_For_Stakeholders.md`
- `docs/Journey_Approach_Validation.md`
- `docs/3_Exact_Tokenization_and_Sequences.md`
- `docs/4_Clustering_Two_Routes.md`
- `docs/7_Cluster_Noise_Postprocessing.md`
- `docs/9_Semantic_Enrichment.md`

## Tests

After the package import layout is restored:

```powershell
python -m unittest discover -s tests -v
```

The tests cover chronological session splitting, canonical production input,
semantic enrichment, token/backoff behavior, journey features, cluster assignment,
scoring, and hierarchical post-processing.
