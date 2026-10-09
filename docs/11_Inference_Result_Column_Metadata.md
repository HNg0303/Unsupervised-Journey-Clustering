# Journey inference result: column metadata

This document defines the columns in the scored journey inference result. It
applies to the partitioned parquet files under `output/scores/` and to the
merged `<platform>_scores.csv` written by `scripts/run_full_pipeline.sh`. The
columns and their order are those of the `journey_summary` database table
(`OUTPUT_COLUMNS` in `src/journey_clustering/cli/infer.py`).

The logical grain is **one row per journey**. A session may contain multiple
journeys. Business names are not repeated on every row: each journey carries
the HiFPT sitemap `taxonomy_id` of its cluster, and the names live once per id
in `journey_sitemap_taxonomy` (`taxonomy/hifpt_sitemap_taxonomy.csv`).

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
| `taxonomy_id` | string, nullable | HiFPT sitemap id of the cluster's reviewed name: feature `M05.06.03`, submodule `M05.06` or module `M05`. Empty for cluster `-1` and before the clusters are named. |
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
| `cluster_name`, `cluster_name_en`, `business_family`, `business_family_code`, `naming_confidence`, … | `taxonomy_id`, join `journey_sitemap_taxonomy` |

## Business names: HiFPT sitemap taxonomy

`journey_sitemap_taxonomy` holds the HiFPT sitemap taxonomy, one row per
module, submodule and feature (16 + 85 + 486 = 587 ids). It is exported from the
sitemap JSON with `journey-sitemap` into `taxonomy/hifpt_sitemap_taxonomy.csv`.

| Column | Metadata |
|---|---|
| `taxonomy_id` | `M05` (module), `M05.06` (submodule) or `M05.06.03` (feature). Primary key. |
| `business_family` | Module, level 1. |
| `business_submodule` | Submodule, level 2; empty on module rows. |
| `business_detail` | Feature, level 3; empty on module and submodule rows. |

### Per-cluster file `journey_cluster_taxonomy.csv`

One row per `(model_version, platform, cluster_id)` with its `taxonomy_id`,
built by `scripts/build_cluster_taxonomy.py` from the reviewed names exported by
the cluster labeling app (`<platform>_named_clusters.csv`). It is a file of the
model run, not a database table. The script refuses names that do not belong to
the scored model (cluster IDs, journey count per cluster and n-gram evidence
must match the run's shareholder catalog) or to the sitemap: a given
`taxonomy_id` must carry its sitemap names, and a cluster named only to level 1
or 2 gets the module or submodule id. Cluster `-1` keeps the noise name
"Chưa phân loại | Journey hỗn hợp/nhiễu" and no id.

### Putting `taxonomy_id` on journeys

- At scoring time: `journey-infer ... --cluster-taxonomy <run>/journey_cluster_taxonomy.csv`.
- On existing scores (`scripts/map_cluster_names.py`, library
  `journey_clustering.cluster_mapping`), reading 200,000 rows at a time:

```bash
journey-map --scores-run output/scores/678/678_20260929_090904 [--with-names]
# -> <run>/android/android_journeys_named.csv, <run>/ios/ios_journeys_named.csv
```

`--with-names` adds the three sitemap names after `taxonomy_id`. Journeys of
a cluster the file does not know get no id, and names from another
`model_version` are refused. Older exports with a `cluster` column are
accepted. In the database the names are a join on `taxonomy_id`.

## Recommended aggregation fields

Per `(model_version, platform, cluster_id)` or per `taxonomy_id`, then join the names:

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
- Sitemap taxonomy and `taxonomy_id` on journeys: `src/journey_clustering/cluster_mapping.py`, `src/journey_clustering/cli/map_clusters.py`, `src/journey_clustering/cli/sitemap_taxonomy.py`.
