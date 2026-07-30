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

## 6. Original-session findings

`session_id` is treated as the original logging-session identifier and is never overwritten.


**Session event-count quantiles**

| quantile | value |
| --- | --- |
| 0.0 | 1.0 |
| 0.25 | 15.0 |
| 0.5 | 34.0 |
| 0.75 | 87.0 |
| 0.9 | 191.60000000000014 |
| 0.95 | 301.5999999999999 |
| 0.99 | 725.76 |
| 1.0 | 1524.0 |


**Session-span quantiles in seconds**

| quantile | value |
| --- | --- |
| 0.0 | 0.0 |
| 0.25 | 36.663 |
| 0.5 | 257.368 |
| 0.75 | 1732.088 |
| 0.9 | 7563.691000000006 |
| 0.95 | 20235.319599999995 |
| 0.99 | 93387.03088 |
| 1.0 | 321112.548 |


**Original-order timestamp issues**

| invalid_primary_event_time_rows | invalid_or_missing_pair_rows | decreasing_timestamp_rows_before_sort |
| --- | --- | --- |
| 0 | 0 | 6101 |

Sessions with more than one non-null device: 0.
Sessions with more than one non-null customer: 174.

## 7. Inactivity-threshold comparison

Configurable thresholds of 15, 30, and 60 minutes are compared. The selected 30-minute threshold is a modeling assumption for this notebook, not a confirmed business fact.


**Threshold comparison**

| threshold_minutes | gaps_exceeding_threshold | percentage_of_observed_gaps | resulting_clean_sessions_if_gap_only |
| --- | --- | --- | --- |
| 15 | 544 | 0.481 | 1969 |
| 30 | 313 | 0.2767 | 1738 |
| 60 | 173 | 0.1529 | 1598 |


**Clean-session split reasons**

| session_split_reason | count |
| --- | --- |
| inactivity>30m | 313 |
| known_customer_changed | 1 |

## 8. Duplicate-event findings


**Duplicate statistics**

| metric | value |
| --- | --- |
| duplicate_record_id_rows | 0 |
| duplicate_record_id_excess | 0 |
| fully_duplicated_rows | 0 |

## 9. Exact token definition

Each event is represented as the exact tuple `(event_type, segment_name, screen_name)`. For example, `["View","HomeVC",null]`, `["View","android/Home",null]`, `["Action","btn_back","HomeVC"]`, and `["Action","btn_back","android/Home"]` remain four different tokens if observed.

## 10. Why semantic canonicalization is intentionally not performed

Confirmed source fact: distinct segment and screen strings are source-defined distinct components. Therefore, semantic canonicalization would remove information that the source owner says is meaningful. This work performs schema validation, derived timestamp parsing, deterministic exact token ID assignment, and derived missing-value representation only.

## 11. Missing-value handling inside tokens

Original missing source values remain missing. In derived serialized token strings only, missing token fields are represented explicitly as `<MISSING>` inside a JSON array. The sentinel is validated not to collide with observed source values.

## 12. Deterministic token-ID strategy

The exact JSON token strings are sorted deterministically, then assigned integer IDs from 1 to 3338. Python runtime hashes are not used.


**Token dictionary preview**

| token_id | exact_token | event_type | segment_name | screen_name | total_event_frequency | original_session_frequency | clean_session_frequency | first_observed_timestamp | last_observed_timestamp |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | ["Action","/","DeviceDetailVC"] | Action | / | DeviceDetailVC | 1 | 1 | 1 | 2026-07-07 04:38:37.692000+00:00 | 2026-07-07 04:38:37.692000+00:00 |
| 2 | ["Action","/","HomeVC"] | Action | / | HomeVC | 1 | 1 | 1 | 2026-07-02 10:19:07.540000+00:00 | 2026-07-02 10:19:07.540000+00:00 |
| 3 | ["Action","/","ModemRestartScheduleVC"] | Action | / | ModemRestartScheduleVC | 9 | 5 | 5 | 2026-07-01 08:00:32.971000+00:00 | 2026-07-06 04:48:37.295000+00:00 |
| 4 | ["Action","/","ServiceManageVC"] | Action | / | ServiceManageVC | 2 | 2 | 2 | 2026-07-02 10:29:34.188000+00:00 | 2026-07-06 04:42:45.638000+00:00 |
| 5 | ["Action","/Body/BackButton","http://172.20.10.2:5173/web/game/spin-wheel/gift"] | Action | /Body/BackButton | http://172.20.10.2:5173/web/game/spin-wheel/gift | 5 | 3 | 4 | 2026-07-02 09:25:11.918000+00:00 | 2026-07-03 03:19:12.625000+00:00 |
| 6 | ["Action","/Body/BackButton","http://172.20.10.2:5173/web/game/spin-wheel/gift?tab=1"] | Action | /Body/BackButton | http://172.20.10.2:5173/web/game/spin-wheel/gift?tab=1 | 1 | 1 | 1 | 2026-07-02 10:15:53.960000+00:00 | 2026-07-02 10:15:53.960000+00:00 |
| 7 | ["Action","/Body/BackButton","http://172.20.10.2:5173/web/game/spin-wheel/home?env=testing&navigationType=previous"] | Action | /Body/BackButton | http://172.20.10.2:5173/web/game/spin-wheel/home?env=testing&navigationType=previous | 3 | 1 | 1 | 2026-07-02 09:25:23.523000+00:00 | 2026-07-02 09:38:07.151000+00:00 |
| 8 | ["Action","/Body/BackButton","http://172.20.10.2:5173/web/game/spin-wheel/home?env=testing&navigationType=previous&test=0567457187"] | Action | /Body/BackButton | http://172.20.10.2:5173/web/game/spin-wheel/home?env=testing&navigationType=previous&test=0567457187 | 1 | 1 | 1 | 2026-07-02 10:15:55.014000+00:00 | 2026-07-02 10:15:55.014000+00:00 |
| 9 | ["Action","/Body/BackButton","http://172.20.10.2:5173/web/game/spin-wheel/home?env=testing&navigationType=previous&test=2003344"] | Action | /Body/BackButton | http://172.20.10.2:5173/web/game/spin-wheel/home?env=testing&navigationType=previous&test=2003344 | 1 | 1 | 1 | 2026-07-03 03:14:42.328000+00:00 | 2026-07-03 03:14:42.328000+00:00 |
| 10 | ["Action","/Body/BackButton","http://172.20.10.2:5173/web/game/spin-wheel/home?env=testing&navigationType=previous&test=2005444"] | Action | /Body/BackButton | http://172.20.10.2:5173/web/game/spin-wheel/home?env=testing&navigationType=previous&test=2005444 | 1 | 1 | 1 | 2026-07-02 09:43:41.894000+00:00 | 2026-07-02 09:43:41.894000+00:00 |
| 11 | ["Action","/Body/BackButton","http://172.20.10.2:5173/web/game/spin-wheel/mission"] | Action | /Body/BackButton | http://172.20.10.2:5173/web/game/spin-wheel/mission | 5 | 3 | 4 | 2026-07-02 09:25:03.219000+00:00 | 2026-07-03 03:14:01.259000+00:00 |
| 12 | ["Action","/Body/BackButton","https://staging-hi.fpt.vn/web/game/game-checkin/home?utm_source=banner"] | Action | /Body/BackButton | https://staging-hi.fpt.vn/web/game/game-checkin/home?utm_source=banner | 5 | 4 | 4 | 2026-07-01 03:47:02.634000+00:00 | 2026-07-06 10:06:46.254000+00:00 |
| 13 | ["Action","/Body/BackButton","https://staging-hi.fpt.vn/web/game/spin-wheel/gift"] | Action | /Body/BackButton | https://staging-hi.fpt.vn/web/game/spin-wheel/gift | 71 | 33 | 34 | 2026-07-01 02:58:14.109000+00:00 | 2026-07-06 07:58:59.408000+00:00 |
| 14 | ["Action","/Body/BackButton","https://staging-hi.fpt.vn/web/game/spin-wheel/gift?tab=1"] | Action | /Body/BackButton | https://staging-hi.fpt.vn/web/game/spin-wheel/gift?tab=1 | 8 | 8 | 8 | 2026-07-01 02:57:38.833000+00:00 | 2026-07-03 10:03:42.001000+00:00 |
| 15 | ["Action","/Body/BackButton","https://staging-hi.fpt.vn/web/game/spin-wheel/home?utm_source=banner"] | Action | /Body/BackButton | https://staging-hi.fpt.vn/web/game/spin-wheel/home?utm_source=banner | 101 | 43 | 49 | 2026-07-01 02:58:14.904000+00:00 | 2026-07-06 08:59:54.219000+00:00 |
| 16 | ["Action","/Body/BackButton","https://staging-hi.fpt.vn/web/game/spin-wheel/home?utm_source=noti"] | Action | /Body/BackButton | https://staging-hi.fpt.vn/web/game/spin-wheel/home?utm_source=noti | 13 | 12 | 12 | 2026-07-01 14:36:54.724000+00:00 | 2026-07-06 03:57:35.859000+00:00 |
| 17 | ["Action","/Body/BackButton","https://staging-hi.fpt.vn/web/game/spin-wheel/introduce"] | Action | /Body/BackButton | https://staging-hi.fpt.vn/web/game/spin-wheel/introduce | 34 | 16 | 16 | 2026-07-01 02:23:11.222000+00:00 | 2026-07-06 02:31:49.475000+00:00 |
| 18 | ["Action","/Body/BackButton","https://staging-hi.fpt.vn/web/game/spin-wheel/list-redeem"] | Action | /Body/BackButton | https://staging-hi.fpt.vn/web/game/spin-wheel/list-redeem | 35 | 15 | 17 | 2026-07-01 02:22:33.781000+00:00 | 2026-07-06 03:57:26.490000+00:00 |
| 19 | ["Action","/Body/BackButton","https://staging-hi.fpt.vn/web/game/spin-wheel/mission"] | Action | /Body/BackButton | https://staging-hi.fpt.vn/web/game/spin-wheel/mission | 124 | 46 | 54 | 2026-07-01 02:24:11.892000+00:00 | 2026-07-06 08:59:16.686000+00:00 |
| 20 | ["Action","/Body/ChangeTabGift","http://172.20.10.2:5173/web/game/spin-wheel/gift"] | Action | /Body/ChangeTabGift | http://172.20.10.2:5173/web/game/spin-wheel/gift | 12 | 3 | 4 | 2026-07-02 09:25:05.607000+00:00 | 2026-07-03 03:19:11.955000+00:00 |

## 13. Raw versus compressed session sequences

Raw sequences retain every event token. Compressed sequences use run-length encoding for consecutive exact-token repeats and never exceed the raw length.


**Original-sequence compression statistics**

| metric | value |
| --- | --- |
| mean_compression_ratio_compressed_to_raw | 0.8336797136842106 |
| median_compression_ratio_compressed_to_raw | 0.864865 |
| sessions_with_repeated_exact_tokens | 1350.0 |


**Representative original sequences**

| example_type | session_id | raw_event_count | compressed_event_count | compression_ratio_compressed_to_raw | raw_token_sequence_preview | readable_sequence_preview |
| --- | --- | --- | --- | --- | --- | --- |
| short_session | 67be4e4e-bc26-4e9b-b84b-ccb3611d1132 | 1 | 1 | 1.0 | [3330] | ['["View","loylaty_promotion_view","<MISSING>"]'] |
| median_length_session | 3F409BEE-1A91-4FC6-ABEE-676C31F76FB6 | 34 | 32 | 0.941176 | [2668, 2668, 2606, 2587, 2642, 2642, 1252, 259, 2664, 2606, 2587, 1147, 2607, 2664, 1153, 2613, 2607, 1156, 2610, 2613] | ['["View","SplashVC","<MISSING>"]', '["View","SplashVC","<MISSING>"]', '["View","MainTabBarController","<MISSING>"]', '["View","HomeVC","<MISSING>"]', '["View","PopupBigMessageVC","<MISSING>"]', '["View","PopupBigMessageVC","<MISSING>"]', '["Action","invite_update_close","HomeVC"]', '["Action","Home/other_manage_internet","HomeVC"]', '["View","ServiceManageVC","<MISSING>"]', '["View","MainTabBarController","<MISSING>"]', '["View","HomeVC","<MISSING>"]', '["Action","home/home_service_management/internet_service_tab/click_modem_control","ServiceManageVC"]', '["View","ManageModemVC","<MISSING>"]', '["View","ServiceManageVC","<MISSING>"]', '["Action","home/home_service_management/internet_service_tab/modem_control/click_modem_schedule","ManageModemVC"]', '["View","ModemRestartScheduleVC","<MISSING>"]', '["View","ManageModemVC","<MISSING>"]', '["Action","home/home_service_management/internet_service_tab/modem_control/modem_schedule/add_modem_schedule","ModemRestartScheduleVC"]', '["View","ModemCreateRestartScheduleVC","<MISSING>"]', '["View","ModemRestartScheduleVC","<MISSING>"]', '... (14 more)'] |
| long_session | D01CBD18-09A0-460E-A863-62CDC42FE1ED | 1524 | 1368 | 0.897638 | [2693, 3217, 2686, 2693, 3217, 2686, 2668, 2526, 2603, 2682, 2668, 2686, 2682, 1261, 1255, 1258, 2655, 2656, 2700, 2655] | ['["View","WebkitEcommerceController","<MISSING>"]', '["View","https://staging-hi.fpt.vn/dkol/update-package/home?merchant_id=BH_UINT&contractNo=SGABP2073&navigationType=previous","<MISSING>"]', '["View","UITrackingElementWindowController","<MISSING>"]', '["View","WebkitEcommerceController","<MISSING>"]', '["View","https://staging-hi.fpt.vn/dkol/update-package/home?merchant_id=BH_UINT&contractNo=SGABP2073&navigationType=previous","<MISSING>"]', '["View","UITrackingElementWindowController","<MISSING>"]', '["View","SplashVC","<MISSING>"]', '["View","BaseNavigation","<MISSING>"]', '["View","LoginVC","<MISSING>"]', '["View","UIHostingController<PopupView>","<MISSING>"]', '["View","SplashVC","<MISSING>"]', '["View","UITrackingElementWindowController","<MISSING>"]', '["View","UIHostingController<PopupView>","<MISSING>"]', '["Action","login_button_clear","LoginVC"]', '["Action","login/continue_login","LoginVC"]', '["Action","login/login_with_fid","LoginVC"]', '["View","SFAuthenticationViewController","<MISSING>"]', '["View","SFBrowserRemoteViewController","<MISSING>"]', '["View","_UISceneHostingViewController","<MISSING>"]', '["View","SFAuthenticationViewController","<MISSING>"]', '... (1504 more)'] |
| session_with_repeated_exact_tokens | 60f6a750-e80b-4503-8331-d08abcffae99 | 1293 | 1068 | 0.825986 | [2725, 2667, 2667, 2605, 2605, 2666, 2666, 2573, 2573, 2705, 471, 2705, 2575, 2575, 3265, 31, 23, 3265, 3262, 39] | ['["View","android/home/payment","<MISSING>"]', '["View","SplashActivity","<MISSING>"]', '["View","SplashActivity","<MISSING>"]', '["View","MainAppActivity","<MISSING>"]', '["View","MainAppActivity","<MISSING>"]', '["View","Splash","<MISSING>"]', '["View","Splash","<MISSING>"]', '["View","HOME","<MISSING>"]', '["View","HOME","<MISSING>"]', '["View","android/Home","<MISSING>"]', '["Action","banner","android/Home"]', '["View","android/Home","<MISSING>"]', '["View","HiWebViewActivity","<MISSING>"]', '["View","HiWebViewActivity","<MISSING>"]', '["View","https://staging-hi.fpt.vn/web/game/spin-wheel/home?utm_source=banner","<MISSING>"]', '["Action","/Body/GoToPageButton/gift","https://staging-hi.fpt.vn/web/game/spin-wheel/gift"]', '["Action","/Body/ChangeTabGift","https://staging-hi.fpt.vn/web/game/spin-wheel/gift"]', '["View","https://staging-hi.fpt.vn/web/game/spin-wheel/home?utm_source=banner","<MISSING>"]', '["View","https://staging-hi.fpt.vn/web/game/spin-wheel/gift","<MISSING>"]', '["Action","/Body/HandleBack","https://staging-hi.fpt.vn/web/game/spin-wheel/update-information?item_id=1625&item_type=VALUABLE&id=461297&"]', '... (1273 more)'] |
| session_with_missing_token_fields | 000A5106-BBC0-45BA-AC04-53BAF35B99F3 | 31 | 29 | 0.935484 | [2668, 2668, 2606, 2585, 932, 2519, 2585, 904, 2526, 2603, 2606, 2519, 1255, 1259, 2622, 2603, 2622, 2606, 2587, 210] | ['["View","SplashVC","<MISSING>"]', '["View","SplashVC","<MISSING>"]', '["View","MainTabBarController","<MISSING>"]', '["View","HomeGuestVC","<MISSING>"]', '["Action","guest/Home/Nav_profile","HomeGuestVC"]', '["View","AccountHomeController","<MISSING>"]', '["View","HomeGuestVC","<MISSING>"]', '["Action","guest/Account-Login-Guest/Buttons/Click_login","AccountHomeController"]', '["View","BaseNavigation","<MISSING>"]', '["View","LoginVC","<MISSING>"]', '["View","MainTabBarController","<MISSING>"]', '["View","AccountHomeController","<MISSING>"]', '["Action","login/continue_login","LoginVC"]', '["Action","login/login_with_otp","LoginVC"]', '["View","OTPCreatePinVC","<MISSING>"]', '["View","LoginVC","<MISSING>"]', '["View","OTPCreatePinVC","<MISSING>"]', '["View","MainTabBarController","<MISSING>"]', '["View","HomeVC","<MISSING>"]', '["Action","Home/Nav_support","HomeVC"]', '... (11 more)'] |

Saved output files are written under `outputs/`. The token dictionary is available as CSV and JSON Lines; session sequences are JSON Lines to preserve nested arrays and compressed repeat objects.


**Saved tokenized output files**

| name | path | rows |
| --- | --- | --- |
| token_dictionary.csv | D:\BehaviourClassification\outputs\token_dictionary.csv | 3338 |
| token_dictionary.jsonl | D:\BehaviourClassification\outputs\token_dictionary.jsonl | 3338 |
| original_session_sequences.jsonl | D:\BehaviourClassification\outputs\original_session_sequences.jsonl | 1425 |
| clean_session_sequences.jsonl | D:\BehaviourClassification\outputs\clean_session_sequences.jsonl | 1739 |
| tokenized_output_manifest.json | D:\BehaviourClassification\outputs\tokenized_output_manifest.json | 1 |

## 14. Validation results

Validation passed for row preservation, record traceability, one-to-one token mapping, reversible serialization, unchanged source columns, deterministic ordering, session count reconciliation, sequence lengths, compressed sequence lengths, repeated dictionary determinism, and unchanged input file fingerprint.


**Validation summary**

| source_rows | tokenized_rows | token_vocabulary_size | original_sequence_count | clean_sequence_count | validation_status |
| --- | --- | --- | --- | --- | --- |
| 114534 | 114534 | 3338 | 1425 | 1739 | passed |

## 15. Limitations and unresolved questions

Unresolved questions:

- Whether the 30-minute inactivity threshold should be changed by business policy.
- Whether timestamp and `created_at` disagreement has source-system meaning.
- Whether duration units and extreme duration semantics are formally documented by the source owner.
- Whether missing `segment_name` or `screen_name` values are structural for specific event types.

## 16. Recommended next step toward journey-candidate discovery

The next step should be journey-candidate discovery using the validated exact-token session sequences. That next phase can compare candidate mining approaches, but this deliverable intentionally stops before PrefixSpan, embeddings, clustering, journey labeling, and cluster evaluation.

## Classification of statements

- Confirmed source facts: distinct `segment_name` and `screen_name` values are distinct components or segments.
- Observed data evidence: all counts, distributions, timestamp comparisons, duration summaries, duplicate findings, and token frequencies above.
- Modeling assumptions: derived clean-session split conditions and the selected 30-minute inactivity threshold.
- Configurable thresholds: 15, 30, and 60 minute inactivity thresholds, with 30 minutes selected for derived clean sessions.
- Derived fields: parsed datetime columns, numeric duration fields, missingness flags, clean-session fields, exact token strings, token IDs, neighbor token context, and sequence metrics.
