# Journey Clustering dashboard

The Streamlit app is a presentation layer over prepared artifacts. It does not run the
large raw-event or inference scans itself.

## Sources by page

| Page | Source of truth | Grain |
| --- | --- | --- |
| Overview | train run metadata + `html_dashboard_summary/kpi.csv` | inference journey |
| Raw data / EDA | `post_analysis/eda/eda_summary.json` | raw event CSV |
| Training outputs | `output/partitioned_runs/.../latest/{android,ios}` | train/holdout journey |
| Learned journey types | train cluster catalog + `Cluster_naming.csv` | platform + cluster |
| Production results | `html_dashboard_summary/*.csv` | full-data scored journey |
| Customer results | `shareholder_analysis/*.csv` | customer / action record |

## Preparation order

```bash
python scripts/prepare_data_for_post_analysis/eda_raw.py --input-root data/giga_data --max-rows 1000 \
  --output-dir output/scores/pca48_ngrams12_500/post_analysis/eda  # smoke test

python scripts/prepare_data_for_post_analysis/inference_summary.py \
  --android output/.../android_all_named.csv \
  --ios output/.../ios_all_named.csv \
  --naming output/scores/pca48_ngrams12_500/Cluster_naming.csv \
  --output-dir output/scores/pca48_ngrams12_500/html_dashboard_summary

python dashboard/prepare_dashboard_data.py
python -m streamlit run dashboard/app.py
```

Omit `--max-rows` when running the full raw EDA. If that file is absent, the EDA page
shows an explicit “not prepared” state and the train/inference pages remain usable; it
does not display prepared-Parquet footer counts as raw-event statistics.

## Inference summary contract

`html_dashboard_summary/kpi.csv`, cluster/family tables, friction tables, daily trends and
next-action tables are generated out-of-core from `*_all_named.csv`. `journey_examples.csv`
is a bounded audit sample only. Names are rejoined from the authoritative
`Cluster_naming.csv` using `(platform, cluster_id)`.
