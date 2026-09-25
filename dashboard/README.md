# Journey Clustering dashboard

The Streamlit app is a presentation layer over prepared artifacts. It does not run the
large raw-event or inference scans itself.

## Sources by page

| Page | Source of truth | Grain |
| --- | --- | --- |
| Overview | train run metadata + `html_dashboard_summary/kpi.csv` | inference journey |
| Raw data / EDA | bundle-specific `post_analysis/eda/eda_summary.json` when available + journey summaries | raw event / inferred journey |
| Training outputs | `output/partitioned_runs/.../latest/{android,ios}` | train/holdout journey |
| Learned journey types | bundle training catalog + platform cluster mapping + compact inference summaries | platform + cluster + timeframe |
| Production results | discoverable `output/scores/<run>/html_dashboard_summary/*.csv` | full-data scored journey; selectable run/platform/timeframe |
| Customer results | `shareholder_analysis/*.csv` | customer / action record |

## Launch

```bash
python dashboard/prepare_dashboard_data.py
python -m streamlit run dashboard/app.py
```

Dashboard summaries and raw-EDA files are optional precomputed artifacts, not
core pipeline scripts. If raw EDA is absent, the page shows journey-level EDA
and clearly marks the raw-event scan as unavailable.

The Details pages share the inference bundle, available platform and timeframe controls.
Monthly bundles can contain only one platform (for example July iOS); unavailable
platforms are not offered as empty selections.

## Inference summary contract

`html_dashboard_summary/kpi.csv`, cluster/family tables, friction tables, daily trends and
next-action tables are generated out-of-core from `*_all_named.csv`. Date-grain tables
(`daily_kpi.csv`, `daily_cluster_summary.csv`, `daily_family_summary.csv`,
`daily_friction_summary.csv`, `daily_friction_by_type.csv`, and
`daily_next_action_summary.csv`) power the timeframe selector. `journey_examples.csv`
is a bounded audit sample only; the event inspector can resolve any journey id to its
complete ordered `sequence` from the scored export. Names are rejoined from the supplied
cluster mapping using `(platform, cluster_id)`.
