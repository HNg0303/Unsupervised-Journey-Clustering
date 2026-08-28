# Large-Data Processing and Continuous-Learning Plan

## 1. Objective and conclusion

Scale the journey-clustering system from roughly 500 MB inputs to raw CSV files
of 10 GB or more, while supporting newly arriving data, continuous inference,
periodic model adaptation, reproducibility, and rollback.

The target flow is:

```text
Raw CSV
  -> chunked validation and canonicalization
  -> partitioned canonical-event Parquet
  -> session-safe journey construction
  -> partitioned journey Parquet
  -> A. sequential inference with a frozen production model
  -> B. periodic candidate-model training and validation
```

Do not concatenate every Parquet file into one in-memory DataFrame. The files
together form one logical dataset, read by partition or bounded batch.

The journey and scoring definitions can remain mostly unchanged. The major work
is adding an out-of-core data layer and separating inference from training.
TF-IDF, SVD, and HDBSCAN are currently batch estimators, so Parquet alone does
not make the model incremental.

## 2. Why the current pipeline cannot simply read a 10 GB file

A 10 GB CSV can expand several times in memory because of pandas strings,
intermediate copies, token columns, Python sequence lists, feature matrices, and
clustering artifacts. The main limitations are:

1. Chunked raw frames are eventually retained and combined.
2. Prepared platform data is loaded as a complete DataFrame.
3. All journey sequences and feature blocks are retained during training.
4. HDBSCAN fits the complete dense journey embedding and has no `partial_fit`.
5. Large CSV outputs duplicate data and are expensive to scan again.
6. Independently processing arbitrary chunks can split sessions and corrupt
   screen context, event gaps, journey boundaries, and sequences.

The system therefore needs bounded stages with durable outputs between them.

## 3. Proposed storage layout

Use layered datasets locally or in object storage:

```text
data/lake/
  manifests/
    ingestion_manifest.parquet
    journey_manifest.parquet
    inference_manifest.parquet

  canonical_events/
    event_date=2026-08-01/
      platform=android/
        bucket=003/part-....parquet

  pending_sessions/
    processing_date=2026-08-01/
      bucket=003/part-....parquet

  journeys/
    journey_start_date=2026-08-01/
      platform=android/part-....parquet

  scores/
    model_version=2026-08-14T020000Z/
      score_date=2026-08-15/
        platform=android/part-....parquet

  training_sets/
    training_set_id=.../
      manifest.json
      journey_sample.parquet

models/
  android/2026-08-14T020000Z/
    journey_scorer.pkl
    config.json
    metrics.json
    training_manifest.json
    cluster_identity_mapping.json
```

Partition primarily by date and platform. To keep all events from one session
together, derive a fixed bucket:

```text
session_bucket = stable_hash(platform + session_id) % 128
```

Do not partition by individual session ID. That creates too many small files.
Aim for Parquet files around 128-512 MB and compact small files periodically.

## 4. Data contracts

### 4.1 Raw-source manifest

Store one record per input file:

```text
source_uri, source_size, source_modified_time, source_checksum
ingestion_run_id, status, schema_version
rows_read, rows_accepted, rows_rejected
minimum_event_time, maximum_event_time, error_message
```

The checksum or equivalent immutable identity makes ingestion idempotent. A
rerun must not duplicate an already completed source.

### 4.2 Canonical events

Reuse the contract in `src/production.py`, then add operational fields:

```text
event_date, session_bucket, ingestion_run_id, schema_version
```

Persist typed UTC timestamps, integer row positions, dictionary-encoded strings
where appropriate, and real nulls rather than textual `nan` values.

### 4.3 Completed journeys

One row represents one completed journey:

```text
journey_id, session_id, device_id, customer_id, platform
journey_start_time, journey_end_time, journey_start_date, boundary_reason
sequence, coarse_sequence, intent_sequence, operation_sequence
n_events_final, n_unique_tokens, action_ratio, back_rate, revisit_ratio
n_loop_removed, n_dedup_removed, span_seconds
median_gap_s, p90_gap_s, max_gap_s
source_min_event_time, source_max_event_time, journey_schema_version
```

Store sequences as Parquet `list<string>` columns. Only join them into display
strings for small reports. This journey dataset is the durable interface between
event processing and modeling; historical raw data should not be rebuilt for
every training or inference run.

## 5. Stage A: raw CSV to canonical-event Parquet

For each unseen input file:

1. Add a `running` record to the ingestion manifest.
2. Read a bounded chunk, initially 100,000-500,000 rows.
3. Normalize aliases and validate required columns.
4. Parse timestamps and normalize platform/event values.
5. Generate or validate stable `record_id` values.
6. Save rejected rows and explicit rejection reasons.
7. Derive `event_date` and `session_bucket`.
8. Write accepted rows immediately to staging Parquet.
9. Release the chunk before reading the next one.
10. Deduplicate, externally sort, and compact staged partitions.
11. Publish outputs atomically, then mark the source `complete`.

Write first to a run-specific staging location. A failed process must never
expose a half-complete partition as production-ready.

Deduplicate by trustworthy `record_id`. Retain a deterministic fallback key
based on platform, session, event time, event type, segment, and source identity.
Define whether identical records in different files are retries or valid repeats.

Required event order is:

```text
platform, session_id, event_time, source_file, source_row_number
```

Use DuckDB external sorting or bounded hash buckets rather than a global pandas
sort. DuckDB is a sensible first implementation because it scans CSV/Parquet and
can spill sorting and grouping work to disk.

## 6. Stage B: session-safe journey construction

A storage chunk, file, or date partition is not a journey boundary. A session
may start in one chunk and end in the next.

For each platform and session bucket:

1. Load the current interval plus unresolved session tails from the prior one.
2. Sort by session and deterministic event order.
3. Apply existing screen-context, semantics, tokens, segmentation, entropy, and
   postprocessing rules to bounded groups of sessions.
4. Emit journeys known to be complete.
5. Persist unresolved session tails for the next interval.
6. Release processed data before opening the next bucket.

Use an event-time watermark:

```text
watermark = maximum_observed_event_time - allowed_lateness
```

A session can be finalized once its last event is older than the watermark and
the idle-gap condition is satisfied. Choose allowed lateness from measured data
delivery delays. Define a policy for events arriving after finalization:

- rebuild and replace affected session journeys;
- write a correction record; or
- quarantine arrivals beyond the supported limit.

For a completed historical import, explicitly flush all pending sessions after
the last partition.

Persist at least the following state for unfinished sessions:

```text
platform, session_id, last_event_time, current_screen_context
events_since_boundary, partial events or sufficient derived state
last processed source identity
```

Initially this can be pending-session Parquet. A future real-time system could
use Redis, RocksDB, or a streaming engine's state store.

Reuse the business logic in `src/production.py`, `src/canonize.py`,
`src/semantics.py`, `src/tokens.py`, `src/segment.py`, and `src/postprocess.py`.
If APIs need changing, add support for already-sorted complete-session batches
without changing journey semantics.

## 7. Stage C: sequential inference

Inference always uses an immutable, versioned production model:

1. Load the production scorer once per worker.
2. Read one bounded completed-journey batch.
3. Pass metadata and sequence channels to `score_prepared`.
4. Write assignments, distances, likelihoods, anomalies, and model version.
5. Mark that input partition scored for that version.
6. Release memory and continue.

Never call `fit_transform` during inference. Use the fitted vocabulary, SVD,
numeric scaler, centroids, Markov bank, and thresholds.

Recommended output fields:

```text
journey_id, model_version, stable_cluster_id, raw_cluster
assignment_type, distance_to_centroid, distance_limit, markov_logprob
geometric_anomaly, generative_anomaly, friction_flags
next_action, scored_at
```

Make `(journey_id, model_version)` idempotent. This also permits shadow scoring
and comparison of multiple models without overwriting older results.

## 8. Stage D: model training

### Recommended first strategy

Keep the existing TF-IDF, SVD, HDBSCAN, Markov, and scorer logic, but train on a
bounded reproducible journey dataset rather than all raw events.

Example training population:

```text
recent 30-90 days of journeys
+ a reservoir sample from older history
+ extra rare-token and rare-family journeys
+ extra previously unassigned/anomalous journeys
```

First measure the journey-level dataset. If the selected journeys and feature
matrix fit safely in memory, load them all. Otherwise train on a deterministic
sample, perhaps initially 200,000-1,000,000 journeys, and tune from measured
memory, runtime, and cluster stability. This is a capacity range, not a fixed
requirement.

After discovery on the training sample, score all other journey partitions in
batches. HDBSCAN does not need to fit every production journey for all journeys
to receive centroid-based assignments and anomaly scores.

Persist a training manifest containing selected partitions and checksums, time
window, sampling rules and seed, before/after counts, schema versions, pipeline
configuration, and code revision.

### Incremental-update limits

| Component | Current incremental support | Plan |
|---|---:|---|
| TF-IDF vocabulary/IDF | No clean `partial_fit` | Refit candidate |
| Truncated SVD | No current `partial_fit` | Refit candidate |
| HDBSCAN | No `partial_fit` | Refit candidate |
| Centroids | Technically possible | Freeze per version |
| Markov counts | Can be accumulated | Rebuild candidate initially |
| Monitoring counters | Yes | Update every batch |
| Quantile estimates | Possible | Freeze model thresholds initially |

Do not mutate the deployed HDBSCAN pipeline after every batch. Independently
train a candidate model and replace the production model only after validation.

If true minute-by-minute parameter learning later becomes mandatory, evaluate a
separate architecture such as `HashingVectorizer -> incremental projection ->
MiniBatchKMeans -> online transition counts`. That changes model behavior and
loses some HDBSCAN noise and arbitrary-shape clustering benefits.

## 9. Champion/challenger continuous learning

```text
Champion
  immutable; continuously scores new completed journeys

Challenger
  periodically trains on recent + replay journeys
  shadow-scores a common validation population
  is promoted only after passing gates
```

Retrain on a schedule, for example weekly, and optionally when drift persists.
A drift trigger starts candidate training; it does not authorize deployment.

Promotion gates should cover:

- chronological holdout results;
- assigned/unassigned rates overall and by platform/app version;
- noise share and usable cluster count;
- resampling stability;
- important business-family coverage;
- anomaly-rate plausibility;
- inference latency, memory, and artifact size;
- direct champion/challenger comparison on identical journeys;
- review of new, split, merged, and retired high-volume clusters.

Promotion should atomically change a small current-model pointer. Keep the prior
version for immediate rollback.

## 10. Stable cluster identities

HDBSCAN integer labels are local to one run. Cluster `7` today can be unrelated
to cluster `7` after retraining.

Match challenger clusters to champion clusters using medoid sequence similarity,
top n-gram overlap, entry/exit tokens, behavioral summaries, business-family
compatibility, and assignment overlap on a shared reference set. Centroid-only
matching is unsafe when TF-IDF/SVD is independently refitted because the feature
spaces may differ.

Classify each relationship as continued, new, split, merged, or retired. Expose
a stable business identifier downstream and retain raw labels for diagnostics.
Require review for low-confidence mappings.

## 11. Monitoring

### Ingestion

- discovered/completed/rejected/duplicate files;
- read/accepted/rejected row counts;
- schema changes and missing columns;
- invalid timestamp/session rates;
- event-time range and delivery delay;
- duplicate rate, file sizes, and small-file count.

### Journey construction

- events and journeys per session;
- short-journey drop rate;
- boundary-reason distribution;
- length, span, gap, loop, back, and revisit distributions;
- pending-session count and oldest pending event;
- late-event corrections;
- unknown semantic/token rates.

### Model drift

- unassigned rate;
- new/out-of-vocabulary token rate;
- nearest-centroid distance distribution;
- Markov log-probability distribution;
- cluster population distribution;
- anomaly and friction rates;
- breakdown by platform, app version, and date.

Example starting triggers, to calibrate using real history:

```text
unassigned rate rises by more than 5 percentage points
new-token rate exceeds 3%
cluster population PSI exceeds 0.2
median centroid distance rises by more than 20%
weekly retraining is due
```

Require drift to persist for multiple windows unless it indicates schema or
ingestion failure.

## 12. Proposed code organization

Keep modeling rules in `src/` and introduce orchestration separately:

```text
src/large_data/
  schemas.py              schema/version definitions
  manifests.py            idempotent run and partition manifests
  storage.py              Parquet staging, commit, and batch readers
  ingest.py               bounded CSV ingestion
  partition.py            date/platform/session bucket derivation
  session_state.py        pending sessions and watermarks
  journey_builder.py      canonical partitions -> complete journeys
  inference.py            model-versioned batch scoring
  training_sample.py      rolling-window and replay sampling
  model_registry.py       version, promotion, and rollback
  monitoring.py           quality and drift metrics

scripts/
  ingest_raw_events.py
  build_journey_partitions.py
  score_journey_partitions.py
  build_training_sample.py
  train_candidate_model.py
  validate_candidate_model.py
  promote_model.py
  compact_parquet_partitions.py
```

Every command should accept explicit input/output roots, run ID, platform/date
or partition range, and batch/memory limits. It should produce a machine-readable
run report and nonzero exit status when validation fails.

## 13. Testing requirements

### Equivalence

Run the existing all-at-once path and new partitioned path on one fixture. They
must produce equivalent canonical fields, boundaries, IDs, sequences, numeric
features, and scores within numeric tolerance.

### Boundary fixtures

Test sessions contained in one chunk, crossing chunks, crossing files, crossing
midnight, having equal timestamps, arriving out of order, arriving late, and
containing duplicates. Test exact idle-gap and maximum-length thresholds.

### Recovery and idempotency

- ingestion retries do not duplicate events;
- staging failures do not publish incomplete partitions;
- late-session rebuilding deterministically replaces affected journeys;
- repeated scoring does not duplicate `(journey_id, model_version)`;
- promotion can be rolled back;
- manifests never report completion before outputs validate.

### Scale

Measure peak memory, disk, rows/second, journey throughput, training time,
scoring throughput, and file distribution at increasing volumes. The first key
proof is that ingestion and journey construction memory remain bounded as total
input grows.

## 14. Implementation sequence

### Phase 0: baseline

Capture current output metrics and hashes for a small fixture. Measure raw rows,
sessions, journeys, vocabulary, feature dimensions, runtime, and peak memory.

### Phase 1: canonical Parquet ingestion

Add PyArrow and DuckDB, schemas and manifests, chunked ingestion, stable session
buckets, deduplication, external sorting, compaction, and atomic completion.
Validate progressively on small data, 1 GB, then 10 GB.

### Phase 2: journey materialization

Implement pending-session state and watermark behavior. Generate list-based
journey Parquet and prove equivalence across artificial storage boundaries.

### Phase 3: partitioned inference

Read bounded journey batches, call the scorer's prepared-data route, and write
idempotent model-versioned score partitions with resume/retry behavior.

### Phase 4: bounded candidate training

Implement recent-window plus historical-replay sampling, persist training-set
lineage, reuse the existing model training logic, and validate chronologically.

### Phase 5: production model lifecycle

Add drift reports, champion/challenger gates, cluster identity mapping, atomic
promotion, and rollback. Schedule routine and drift-triggered candidate runs.

### Phase 6: optional streaming

Only add Kafka, Flink, Spark Structured Streaming, Redis, or similar systems if
measured latency cannot be met by hourly/daily bounded jobs. Preserve the same
Parquet and model-version contracts.

## 15. Initial operating parameters

- 32-64 GB RAM and a fast SSD for initial single-node 10 GB work;
- temporary disk capacity of roughly 3-5 times raw input, then refine by measure;
- initial CSV chunks around 250,000 rows;
- 64 or 128 session buckets;
- Snappy for speed or Zstandard for storage efficiency;
- target Parquet files around 128-512 MB;
- hourly or daily ingestion/inference batches;
- weekly candidate training plus drift-triggered runs;
- a 60-day recent window plus historical replay as an initial training policy;
- immutable production models between explicit promotions.

These are starting points, not guarantees. Tune them using row width, session
skew, arrival lateness, journey count, and actual hardware measurements.

## 16. Definition of done

The design is complete when:

1. A 10 GB source ingests with bounded peak memory.
2. Reprocessing produces no duplicate canonical events.
3. Sessions crossing chunks/files/dates match the all-at-once reference output.
4. Journey partitions are scored sequentially with a fixed model.
5. Candidate training uses a reproducible bounded journey set with full lineage.
6. A challenger can be validated, promoted atomically, and rolled back.
7. Stable cluster IDs are mapped across versions instead of exposing raw labels.
8. Data quality, lateness, drift, runtime, memory, and failures are observable.

## 17. Final decision

Implement both layers:

1. Convert raw events into a partitioned, session-safe journey dataset without
   concatenating the complete dataset in memory.
2. Reuse the current model pipeline on a bounded training population, while
   scoring all new journey partitions sequentially with a frozen model.

This retains current journey semantics and scoring behavior while adding the
data organization and model lifecycle needed for large, continuously arriving
datasets.
