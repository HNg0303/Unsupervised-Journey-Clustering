# Journey inference result: column metadata

This document defines the columns in the scored journey inference result. It
applies to the partitioned parquet files under `output/scores/` and to the
concatenated inspection file such as `output/inspection/android_all.parquet`.

The logical grain is **one row per journey**. A session may contain multiple
journeys. The stable inference identity is `(journey_id, model_version)`.

## Important conventions

- Timestamps are UTC-aware timestamps when written by the partitioned scorer.
- Ratios and shares are stored as decimals between `0` and `1`, not percentages.
- `cluster = -1` means the primary scorer did not accept the journey into a
  fitted cluster. For downstream grouping, prefer `effective_cluster_key`.
- `effective_cluster_key = UNKNOWN` means unresolved; values such as `C:21`
  identify the effective cluster namespace and ID.
- `cluster_name`, `cluster_name_en`, and business-family fields are intended to
  be descriptive labels joined from the frozen model mapping. In the currently
  inspected scored files, these fields can contain the unresolved fallback for
  rows that actually have a valid cluster; for reliable reporting, group by
  `effective_cluster_key` and rejoin the frozen mapping before displaying names.
  Cluster IDs are stable only within a model version; names and IDs can change
  after retraining.
- Empty strings and nulls are both possible for optional tokens, predictions,
  identifiers, and friction fields.

## Identity and provenance

| Column | Logical type | Metadata |
|---|---|---|
| `journey_id` | string, required | Unique journey identifier. In partitioned inference it is prefixed with the source partition, for example `platform=android_bucket=000_part-00000::J000002`. Deduplicate using `(journey_id, model_version)` when combining reruns. |
| `session_id` | string, required | Original application session identifier. One session can contain multiple segmented journeys. Use `nunique` for session coverage, not row count. |
| `device_id` | string, nullable | Device identifier inherited from the source data. May be absent or anonymized. Use `nunique` only after handling nulls. |
| `customer_id` | string, nullable | Customer/account identifier inherited from the source data. May be absent, anonymous, or shared across sessions. Treat as sensitive data. |
| `os` | string | Operating system/platform carried by the journey, normally `android` or `ios`. |
| `platform` | string, required | Runtime scoring platform written by the partitioned scorer. Normally matches `os`; use this field to select the platform output. |
| `start_ts` | timestamp UTC | Timestamp of the first event in the journey. |
| `end_ts` | timestamp UTC | Timestamp of the last event in the journey. |
| `scored_at` | timestamp UTC | Time at which the journey was scored and written to the result. This is processing time, not user activity time. |
| `model_version` | string, required | Frozen model/run identifier used for scoring. Always retain this when comparing results across runs. |
| `source_partition` | string | Input partition from which the journey was produced. Useful for lineage, replay, and partition-level monitoring. |

## Journey segmentation and size

| Column | Logical type | Metadata |
|---|---|---|
| `boundary_reason` | string | Reason that the journey started. Typical values are `session_start`, `idle_gap`, `root_return`, `auth_change`, and `length_cap`; empty means no boundary reason was recorded at that row. |
| `n_events_raw` | integer | Number of events in the journey before sequence cleanup. |
| `n_events_final` | integer | Number of events remaining after duplicate/cycle cleanup and configured screen filtering. This is the modelled journey length. |
| `n_unique_tokens` | integer | Number of distinct tokens in the cleaned journey sequence. |
| `n_dropped_screens` | integer | Number of screens/events removed by configured screen filtering, such as OS chrome or boot-screen removal. |
| `n_dedup_removed` | integer | Number of consecutive duplicate events removed during cleanup. |
| `n_loop_removed` | integer | Number of events removed when repeated cycles were collapsed. Values greater than zero indicate detected repeated loops. |

## Behavioural metrics

| Column | Logical type | Range/unit | Metadata |
|---|---|---|---|
| `action_ratio` | float | `[0, 1]` | Fraction of cleaned events that are actions/taps rather than views. Aggregate with a mean or median. |
| `back_rate` | float | `[0, 1]` | Fraction of cleaned events classified as back-navigation actions. Higher values indicate more backtracking. |
| `revisit_ratio` | float | `[0, 1]` | `1 - unique_tokens / n_events_final`. Higher values indicate repeated visits to previously seen screens/tokens. |
| `span_seconds` | float | seconds | Elapsed time from the first to the last event in the journey. This is not necessarily active user time. |
| `median_gap_s` | float | seconds | Median gap between consecutive events in the journey. `0` when there are no gaps. |
| `p90_gap_s` | float | seconds | 90th percentile of consecutive-event gaps. Useful for identifying long pauses. |
| `max_gap_s` | float | seconds | Maximum gap between consecutive events. |

## Journey endpoints and sequence

| Column | Logical type | Metadata |
|---|---|---|
| `entry_token` | string, nullable | First cleaned token in the journey, for example `view@HOME`. Use `value_counts()` for entry-path analysis. |
| `exit_token` | string, nullable | Last cleaned token in the journey. Use `value_counts()` for exit/drop-off analysis. |
| `sequence` | string | Cleaned ordered token path. Tokens are separated by ` -> `. This is a display/export representation; split on that separator only when the token format is known to contain no unescaped separator. |

## Cluster assignment and distance

| Column | Logical type | Metadata |
|---|---|---|
| `cluster` | integer | Primary fitted cluster label. `-1` means rejected as noise/unassigned by the primary scorer after the distance threshold. |
| `nearest_cluster` | integer | Closest fitted cluster before applying the acceptance/distance limit. It can be a valid cluster even when `cluster = -1`. Do not use it as the accepted assignment. |
| `distance_to_centroid` | float | Distance from the journey representation to the selected nearest cluster centroid. Lower is more geometrically similar. |
| `distance_limit` | float | Maximum accepted distance for the assignment. A journey with `distance_to_centroid > distance_limit` becomes geometrically anomalous and normally receives `cluster = -1`. |
| `effective_cluster_key` | string | Downstream grouping key. Examples: `C:21` for a primary C cluster, `B:7` for a secondary B assignment, and `UNKNOWN` for unresolved journeys. |
| `assignment_type` | string | Assignment route. Common values include `C_primary`, `B_secondary_inference`, `B_existing_secondary`, `B_borderline_secondary`, `unassigned_novel`, `rare_recurring_pattern`, and `friction_candidate`. |

For cluster-level aggregation, group by `effective_cluster_key`, not by the
human-readable name. Join names from the matching frozen catalog after grouping.

## Markov likelihood and anomaly signals

| Column | Logical type | Metadata |
|---|---|---|
| `markov_logprob` | float | Length-normalized log-probability of the journey sequence under the assigned/nearest cluster transition model. More negative values indicate less typical transitions. |
| `effective_markov_logprob` | float | Markov score associated with the effective assignment after any hierarchical fallback. Use this for operational reporting. |
| `geometric_anomaly` | boolean | `true` when the journey is too far from the accepted centroid according to the fitted distance threshold. |
| `generative_anomaly` | boolean | `true` when the journey's Markov log-probability is below the fitted generative threshold. |
| `severe_anomaly` | boolean | `true` when both geometric and generative anomaly conditions are met. |

The anomaly booleans are model-relative flags, not probabilities. Report them
as journey shares, for example `mean(geometric_anomaly)`.

## Friction and next-action signals

| Column | Logical type | Metadata |
|---|---|---|
| `friction_flags` | pipe-delimited string, nullable | Full rule-based signal list. Possible flags include `excessive_back`, `navigation_loop`, `screen_thrash`, `slow_journey`, `unknown_archetype`, and `improbable_transitions`. Multiple flags are separated by `|`. |
| `behavioral_friction_flags` | pipe-delimited string, nullable | Operational friction subset after removing assignment/model-diagnostic flags such as `unknown_archetype` and `improbable_transitions`. Use this for user-experience friction reporting. |
| `next_action` | string, nullable | Most likely next token predicted by the fitted Markov model for the journey's effective cluster. Empty when no prediction is available. |
| `next_action_share` | float, nullable | Observed share of the predicted next action among matching transitions. Not a calibrated probability; use as a ranking/support score. |
| `effective_next_action` | string, nullable | Next-action prediction associated with the effective assignment after hierarchical fallback. |
| `effective_next_action_share` | float, nullable | Support/share associated with `effective_next_action`. |

To count friction flags, split the selected column on `|` and explode before
calling `value_counts()`. Do not count the raw string values as individual
flags.

## Human-readable cluster labels

| Column | Logical type | Metadata |
|---|---|---|
| `cluster_name` | string | Vietnamese human-readable name for the effective cluster. The unresolved fallback is `Hành trình chưa phân loại / hỗn hợp`. |
| `cluster_name_en` | string | English human-readable name for the effective cluster. The unresolved fallback is `Unclassified / mixed journeys`. |
| `business_family` | string | Human-readable business-family label emitted by the scorer. In the current Android parquet it is Vietnamese for the unresolved fallback, for example `chưa phân loại`; do not assume the column language is stable across scoring paths. |
| `business_family_code` | string | Stable machine-oriented business-family code, such as `unknown`, `support`, `contracts`, or `device_management`. |
| `naming_confidence` | string | Confidence of the catalog naming decision, normally `high`, `medium`, `low`, `unknown`, or `not_applicable`. This describes label quality, not model assignment confidence. |

## Recommended aggregation fields

For a shareholder-style cluster summary, use the following metrics per
`effective_cluster_key`:

```text
journey_count              = count(journey_id)
journey_share              = journey_count / total journeys
session_count              = nunique(session_id)
device_count               = nunique(device_id)
median_journey_length      = median(n_events_final)
median_duration_seconds    = median(span_seconds)
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

The static representative journey and ranked n-gram evidence should continue
to come from the matching model catalog. The inference parquet contains the
observed journey sequence, but it does not contain the original fitted TF-IDF
vocabulary needed to reproduce the catalog's exact n-gram lift values.

## Source of definitions

- Journey construction and behavioural fields: `src/journey_clustering/postprocess.py`.
- Cluster assignment, anomaly, friction, and next-action fields: `src/journey_clustering/score.py`.
- Partitioned inference metadata: `scripts/score_partitioned_events.py`.
- Training catalog aggregation: `src/journey_clustering/cluster.py`.
- Taxonomy naming and score decoration: `src/journey_clustering/naming.py`, exposed by
  `scripts/taxonomy_cluster_naming_pipeline.py`.
