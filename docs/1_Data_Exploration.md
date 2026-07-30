# Data Exploration and Exact Tokenization

This document was generated from `data/final_clean_events.csv`. The input CSV was not modified.

## 1. Dataset overview

Confirmed source fact: the source owner confirmed that every distinct `segment_name` and `screen_name` is a different component or segment.

Observed data evidence is summarized below. Source values are preserved exactly; no case folding, suffix stripping, URL/path simplification, fuzzy matching, or semantic merging was performed.


**Dataset overview**

| metric | value |
| --- | --- |
| rows | 114534 |
| columns | 14 |
| devices | 182 |
| customers | 154 |
| original_sessions | 1425 |
| clean_sessions | 1739 |
| event_types | 2 |
| distinct_segment_names | 1617 |
| distinct_screen_names | 611 |
| exact_event_tuples | 3338 |
| token_vocabulary_size | 3338 |


**Masked sample**

| record_id | utm_source | device_id | customer_id | session_id | created_at | timestamp | event_type | OS | segment_name | visit | duration | screen_name | segmentation.external_id |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 6a4466...226b | <MISSING> | bf90fc...ad87 | 6793746.0 | 7eb539...e235 | 2026-07-01T00:59:55.014Z | 1782867593723 | View | Android | SplashActivity | 1.0 | 0 | <MISSING> | <MISSING> |
| 6a4466...226c | <MISSING> | bf90fc...ad87 | 6793746.0 | 7eb539...e235 | 2026-07-01T00:59:55.020Z | 1782867593821 | View | Android | SplashActivity | <MISSING> | 0 | <MISSING> | <MISSING> |
| 6a4466...226d | <MISSING> | bf90fc...ad87 | 6793746.0 | 7eb539...e235 | 2026-07-01T00:59:55.014Z | 1782867593823 | View | Android | MainAppActivity | 1.0 | 0 | <MISSING> | <MISSING> |
| 6a4466...716e | <MISSING> | bf90fc...ad87 | 6793746.0 | 7eb539...e235 | 2026-07-01T00:59:55.921Z | 1782867594297 | View | Android | MainAppActivity | <MISSING> | 1 | <MISSING> | <MISSING> |
| 6a4466...ce9a | <MISSING> | bf90fc...ad87 | 6793746.0 | 7eb539...e235 | 2026-07-01T00:59:57.923Z | 1782867594299 | View | Android | Splash | 1.0 | 0 | <MISSING> | <MISSING> |
| 6a4466...270c | <MISSING> | bf90fc...ad87 | 6793746.0 | 7eb539...e235 | 2026-07-01T00:59:57.925Z | 1782867596779 | View | Android | HOME | 1.0 | 0 | <MISSING> | <MISSING> |
| 6a4466...270d | <MISSING> | bf90fc...ad87 | 6793746.0 | 7eb539...e235 | 2026-07-01T00:59:57.924Z | 1782867596941 | View | Android | HOME | <MISSING> | 0 | <MISSING> | <MISSING> |
| 6a4466...270e | <MISSING> | bf90fc...ad87 | 6793746.0 | 7eb539...e235 | 2026-07-01T00:59:57.925Z | 1782867596942 | View | Android | android/Home | 1.0 | 0 | <MISSING> | <MISSING> |

## 2. Data-type findings

The notebooks load source columns as string-backed values for preservation, then create derived numeric and datetime columns. The table below shows pandas' default inferred data types when the CSV is read normally for documentation.


**Inferred data types**

| column | inferred_dtype |
| --- | --- |
| record_id | str |
| utm_source | str |
| device_id | str |
| customer_id | float64 |
| session_id | str |
| created_at | str |
| timestamp | int64 |
| event_type | str |
| OS | str |
| segment_name | str |
| visit | float64 |
| duration | int64 |
| screen_name | str |
| segmentation.external_id | str |

## 3. Missing-value analysis

Missing values are preserved in source columns. Missingness may be structural, but the notebooks do not claim its meaning without source-owner confirmation.


**Missing summary**

| column | missing_count | missing_percentage |
| --- | --- | --- |
| utm_source | 112405 | 98.1412 |
| segmentation.external_id | 110063 | 96.0964 |
| screen_name | 87135 | 76.0778 |
| visit | 71678 | 62.5823 |
| customer_id | 16681 | 14.5642 |
| segment_name | 23 | 0.0201 |
| record_id | 0 | 0.0 |
| device_id | 0 | 0.0 |
| session_id | 0 | 0.0 |
| created_at | 0 | 0.0 |
| timestamp | 0 | 0.0 |
| event_type | 0 | 0.0 |
| OS | 0 | 0.0 |
| duration | 0 | 0.0 |


**Availability by event_type**

| event_type | rows | segment_name_available | segment_name_missing | segment_name_available_pct | screen_name_available | screen_name_missing | screen_name_available_pct | duration_available | duration_missing | duration_available_pct |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Action | 27844 | 27821 | 23 | 99.9174 | 27399 | 445 | 98.4018 | 27844 | 0 | 100.0 |
| View | 86690 | 86690 | 0 | 100.0 | 0 | 86690 | 0.0 | 86690 | 0 | 100.0 |

Rows where both `segment_name` and `screen_name` are missing: 0.
Rows where `segment_name` is present and `screen_name` is missing: 87135.
Rows where `segment_name` is missing and `screen_name` is present: 23.

## 4. Timestamp findings

The `timestamp` unit is determined from evidence rather than assumption. Candidate epoch units are compared to parsed `created_at`, and `ms` is selected.


**Timestamp unit evidence**

| candidate_unit | valid_count | min_parsed | max_parsed | median_abs_diff_vs_created_at_seconds |
| --- | --- | --- | --- | --- |
| s | 114534 | 1970-10-28 14:15:23+00:00 | 1970-09-05 16:08:16+00:00 | 1781202441565.1597 |
| ms | 114534 | 2026-07-01 00:59:53.723000+00:00 | 2026-07-07 04:44:38.896000+00:00 | 1.03 |
| us | 114534 | 1970-01-21 15:14:27.593723+00:00 | 1970-01-21 15:23:19.478896+00:00 | 1781202454.1583545 |
| ns | 114534 | 1970-01-01 00:29:42.867593723+00:00 | 1970-01-01 00:29:43.399478896+00:00 | 1782983656.6115613 |


**Timestamp summary**

| metric | value |
| --- | --- |
| selected_timestamp_unit | ms |
| primary_event_time_column | timestamp_datetime |
| secondary_event_time_column | created_at_datetime |
| timestamp_min | 2026-07-01 00:59:53.723000+00:00 |
| timestamp_max | 2026-07-07 04:44:38.896000+00:00 |
| created_at_min | 2026-07-01 00:59:55.014000+00:00 |
| created_at_max | 2026-07-07 04:44:39.666000+00:00 |
| timestamp_invalid_count | 0 |
| created_at_invalid_count | 0 |
| timestamp_disagreement_median_seconds | 1.03 |
| timestamp_disagreement_99th_seconds | 183.05567000000002 |

Observed ordering choice: `timestamp_datetime` is the primary event-ordering field because timestamp has at least as many valid values and at least as many distinct event times as created_at. The deterministic ordering is `session_id`, primary event time, secondary event time, `record_id`, then source row number.

## 5. Duration findings

Duration is not included in categorical tokens. Derived fields include `duration_numeric`, `log1p_duration`, `duration_missing`, `duration_zero`, and `duration_outlier`. Extreme values are not capped, replaced, or deleted.


**Duration summary**

| metric | value |
| --- | --- |
| missing_duration_count | 0.0 |
| zero_duration_count | 74949.0 |
| negative_duration_count | 0.0 |
| positive_duration_count | 39585.0 |
| minimum | 0.0 |
| maximum | 98982.0 |
| mean | 27.24852882113608 |
| median | 0.0 |


**Duration percentiles**

| quantile | value |
| --- | --- |
| 0.0 | 0.0 |
| 0.25 | 0.0 |
| 0.5 | 0.0 |
| 0.75 | 2.0 |
| 0.9 | 16.0 |
| 0.95 | 47.0 |
| 0.99 | 262.0 |
| 1.0 | 98982.0 |
