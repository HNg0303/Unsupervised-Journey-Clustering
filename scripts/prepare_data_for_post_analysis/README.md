# Post-analysis data preparation

This directory is the boundary between the large row-level outputs and the
dashboard/HTML consumers.  It is intentionally split by grain:

- `eda_raw.py` profiles the original event CSVs. It supports `--max-rows` for
  a smoke test; omit it when running the full EDA.
- `train_output.py` exports small metadata from the fitted train/holdout run.
- `inference_summary.py` is the canonical entrypoint for compact full-inference
  aggregates. It delegates to the DuckDB summary builder and never loads all
  inference rows into Streamlit.
- `build_all.py` writes the preparation contract and optionally runs EDA,
  inference summaries, train metadata and static HTML generation.

The final `*_all_named.csv` files are the anchor for customer and business
post-analysis.  Post-analysis scripts must not read unnamed score partitions or
the raw event lake when an all-named anchor is available.
