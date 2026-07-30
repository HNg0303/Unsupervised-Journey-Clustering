# Session-Based EDA

This document summarizes original-session and derived clean-session exploration. `session_id` is preserved, and `clean_session_id` is derived separately.


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
