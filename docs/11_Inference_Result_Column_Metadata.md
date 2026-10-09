# Journey inference result: column metadata

This document defines the columns in the scored journey inference result. It
applies to the partitioned parquet files under `output/scores/` and to the
merged `<platform>_scores.csv` written by `scripts/run_full_pipeline.sh`. The
columns and their order are those of the `journey_summary` database table
(`OUTPUT_COLUMNS` in `src/journey_clustering/cli/infer.py`).

The logical grain is **one row per journey**. A session may contain multiple
journeys. Cluster names are not repeated on every row: they live once per
cluster in `journey_cluster_taxonomy`, joined on
`(model_version, platform, cluster_id)`.

## Important conventions

- Timestamps are UTC.
- Ratios and shares are stored as decimals between `0` and `1`, not percentages.
- `cluster_id = -1` means the scorer did not accept the journey into a fitted
  cluster. Cluster IDs are stable only within a model version; names and IDs can
  change after retraining.
- `journey_id` restarts with every partition run. Combine runs on
  `(batch date, model_version, journey_id)`.
- Empty strings and nulls are both possible for optional tokens, predictions,
  identifiers, and friction fields.

## Columns

| Column | Logical type | Metadata |
|---|---|---|
| `model_version` | string, required | Frozen model/run identifier used for scoring. |
| `platform` | string, required | `android` or `ios`. |
| `cluster_id` | integer | Fitted cluster label; `-1` = not accepted (too far from every centroid). |
| `journey_id` | string, required | Journey identifier prefixed with its source partition, e.g. `platform=android_bucket=000_part-00000::J000002`. |
| `session_id` | string, required | Application session; one session can contain several journeys. |
| `device_id` | string, nullable | Device identifier. |
| `customer_id` | string, nullable | Customer/account identifier. Sensitive. |
| `start_ts`, `end_ts` | timestamp UTC | First and last event of the journey. Duration = `end_ts - start_ts`. |
| `boundary_reason` | string | Why the journey started: `session_start`, `idle_gap`, `root_return`, `auth_change`, `length_cap`. |
| `n_events_raw` | integer | Events before sequence cleanup. |
| `n_events_final` | integer | Events after duplicate/cycle cleanup and screen filtering; the modelled length. |
| `n_loop_removed` | integer | Events removed when repeated cycles were collapsed; `> 0` means a loop. |
| `action_ratio` | float `[0, 1]` | Share of cleaned events that are actions rather than views. |
| `back_rate` | float `[0, 1]` | Share of cleaned events that are back navigation. |
| `revisit_ratio` | float `[0, 1]` | `1 - unique_tokens / n_events_final`. |
| `median_gap_s`, `p90_gap_s`, `max_gap_s` | float, seconds | Median, 90th percentile and maximum gap between consecutive events. |
| `entry_token`, `exit_token` | string, nullable | First and last cleaned token. |
| `sequence` | string | Cleaned ordered token path, tokens separated by ` -> `. |
| `distance_to_centroid` | float | Distance to the nearest fitted centroid; lower is more similar. |
| `markov_logprob` | float | Length-normalised log-probability of the sequence under the cluster's transition model; more negative is less typical. |
| `geometric_anomaly` | boolean | Too far from every centroid (this is also why `cluster_id = -1`). |
| `generative_anomaly` | boolean | Markov log-probability below the fitted 5th percentile. |
| `severe_anomaly` | boolean | Geometric anomaly and log-probability below the 1st percentile. |
| `friction_flags` | pipe-delimited string, nullable | User-experience friction: `excessive_back`, `navigation_loop`, `screen_thrash`, `slow_journey`. Split on `|` before counting. |
| `next_action` | string, nullable | Most likely next token under the cluster's Markov model. |
| `next_action_share` | float, nullable | Observed share of that next token; a support score, not a calibrated probability. |
| `scored_at` | timestamp UTC | When the partition was scored (processing time). |

## Columns removed from earlier exports

Older bundles carry extra columns. They were dropped because they duplicate or
derive from the columns above:

| Removed | Use instead |
|---|---|
| `os` | `platform` |
| `cluster` | `cluster_id` |
| `span_seconds` | `end_ts - start_ts` |
| `n_unique_tokens` | `round(n_events_final * (1 - revisit_ratio))` |
| `n_dropped_screens`, `n_dedup_removed` | `n_events_raw - n_events_final - n_loop_removed` (both together) |
| `nearest_cluster`, `distance_limit` | model internals; `geometric_anomaly` carries the decision |
| `effective_cluster_key`, `assignment_type` | `cluster_id` (`-1` = unassigned) |
| `effective_markov_logprob`, `effective_next_action`, `effective_next_action_share` | `markov_logprob`, `next_action`, `next_action_share` (single-model copies) |
| `behavioral_friction_flags` | `friction_flags` (now holds the behavioural flags only) |
| `unknown_archetype`, `improbable_transitions` flags | `geometric_anomaly`, `generative_anomaly` |
| `source_partition` | prefix of `journey_id` before `::` |
| `cluster_name`, `cluster_name_en`, `business_family`, `business_family_code`, `naming_confidence`, … | join `journey_cluster_taxonomy` |

## Cluster names: `journey_cluster_taxonomy`

One row per `(model_version, platform, cluster_id)`, built by
`scripts/build_cluster_taxonomy.py` from the reviewed names exported by the
cluster labeling app (`<platform>_named_clusters.csv`). The script refuses names
that do not belong to the scored model: the cluster IDs, the journey count per
cluster and the n-gram evidence must match the run's shareholder catalog.

| Column | Metadata |
|---|---|
| `model_version`, `platform`, `cluster_id` | Join key to the scored journeys. |
| `taxonomy_id` | Business taxonomy leaf, e.g. `M05.06.03`; empty when named only to level 2. |
| `cluster_name` | Full name: level 1 \| level 2 \| level 3. |
| `business_family`, `business_submodule`, `business_detail` | Taxonomy levels 1, 2 and 3. |
| `naming_confidence` | `high`, `medium` or `low`. |
| `naming_source` | How the name was chosen, e.g. `validated_taxonomy_detail`, `business_review`. |
| `needs_review` | `1` when the name still needs a reviewer. |
| `named_at` | When the name was last saved in the labeling app. |

## Recommended aggregation fields

Per `(model_version, platform, cluster_id)`, then join the names:

```text
journey_count              = count(journey_id)
journey_share              = journey_count / total journeys
session_count              = nunique(session_id)
device_count               = nunique(device_id)
median_journey_length      = median(n_events_final)
median_duration_seconds    = median(end_ts - start_ts)
mean_action_ratio          = mean(action_ratio)
mean_back_rate             = mean(back_rate)
mean_revisit_ratio         = mean(revisit_ratio)
loop_journey_share         = mean(n_loop_removed > 0)
geometric_anomaly_share    = mean(geometric_anomaly)
generative_anomaly_share   = mean(generative_anomaly)
severe_anomaly_share       = mean(severe_anomaly)
top_entry_token            = mode(entry_token)
top_exit_token             = mode(exit_token)
```

## Source of definitions

- Journey construction and behavioural fields: `src/journey_clustering/postprocess.py`.
- Cluster assignment, anomaly, friction, and next-action fields: `src/journey_clustering/score.py`.
- Published column set and order: `src/journey_clustering/cli/infer.py`.
- Taxonomy naming per cluster: `src/journey_clustering/naming.py`.
- Reviewed name table: `src/journey_clustering/cli/cluster_taxonomy.py`.
