# Journey Clustering dashboard

Streamlit dashboard over the production run in
`output/journey_runs/EXACT_ch-c45i30o20_ng1-3_svd64_fdf3_mf20000_nw0p35_mcs100_ms5_sel-eom_gap90_jmin4_tdf3_ent0_chr1_boot0_test0p2`.

## Run it

```bash
D:/miniconda3/envs/cluster/python.exe -m pip install streamlit plotly pyarrow
```

```bash
D:/miniconda3/envs/cluster/python.exe dashboard/prepare_dashboard_data.py
```

```bash
D:/miniconda3/envs/cluster/python.exe -m streamlit run dashboard/app.py
```

## Theme

The look is defined natively in [`.streamlit/config.toml`](../.streamlit/config.toml) at the repo
root — no CSS injection, so it survives Streamlit upgrades. It is a light theme: white canvas,
`#F4F6FA` sidebar and widget surfaces, `#2C6BB3` accent, Inter at 15px, 8px radii. Chart colours
in the config mirror `PALETTE` in [lib.py](lib.py) so Plotly figures that pass an explicit colour
sequence and those that do not end up matching.

Because the file uses a single `[theme]` section, the app always renders light regardless of the
viewer's OS setting. To offer a light/dark switch in the app's settings menu instead, rename
`[theme]` → `[theme.light]` and `[theme.sidebar]` → `[theme.light.sidebar]`, then add
`[theme.dark]` / `[theme.dark.sidebar]` sections.

Note the theme only applies when the app is launched from the repo root (Streamlit resolves
`.streamlit/config.toml` against the working directory).

## Language

The sidebar has an **EN / VI** toggle. It switches every label, narrative and chart axis, and
also swaps the data-side labels: cluster names use `cluster_name_vi`, business families use the
Vietnamese labels, and representative journeys come from `shareholder_cluster_catalog_vi.json`.
Technical vocabulary (session, journey, cluster, token, action ratio, back rate, HDBSCAN,
TF-IDF, …) is deliberately kept in English in the Vietnamese copy.

Adding a string: wrap it in `lib.t(en, vi)`. Page keys are language-stable, so switching
language keeps the current page and all filter selections.

## Structure

The sidebar has two top-level sections.

### Overview — for stakeholders

One page ([views/page_overview.py](views/page_overview.py)) that answers "what is this and why is
it hard", with no prior context assumed:

- **Input → output**, side by side on *one real production session* carried end to end
  (`showcase.json`): the raw event rows on the left, the named/measured/scored journeys on the right.
- **Why the textbook recipe does not transfer** — this dataset against the clickstream corpora
  papers benchmark on (MSNBC, YOOCHOOSE, RetailRocket, Taobao UserBehavior). Baseline figures are
  the datasets' own published headline numbers, rounded, and are there to contrast *structure*:
  they all ship curated labels and clean ids; production telemetry ships neither.
- **The five steps** — canonize → tokenize → build journeys → represent → cluster + anomaly
  detection, each in business language with the number it produced on this run.
- **What the business gets** — the four usable outputs.

### Details — the evidence

| Page | Question it answers | Reads |
| --- | --- | --- |
| 1 · Raw data (EDA) | What does production clickstream look like, and why does it need a taxonomy? | `output/dashboard_cache/eda.json` |
| 2 · Training outputs | What did the run write, and what does each artefact say about the model? | `output/dashboard_cache/train_run.json` + the run's `*_report_*.csv` |
| 3 · Learned journey types | What behaviours did the model discover, and what is the evidence for each name? | `*_report_cluster_catalog.csv`, `cluster_name_mapping.csv`, `*_cluster_ngrams.csv`, `shareholder_cluster_catalog.json` |
| 4 · Production results | What did users actually do on the held-out T5 extract — as business outcomes, including how the top journeys trend over time? | `output/dashboard_cache/inference_{platform}.parquet` |

### A note on the T5 time axis

The scored extract runs 2026-03-02 → 2026-06-04, but ~98% of its journeys fall in
2026-04-30 → 2026-05-31; the rest are stray early timestamps. Time-series charts and the
risers/fallers comparison are therefore drawn over that dense window, and the first-vs-second-half
split uses the **median journey** rather than the calendar midpoint (a calendar split puts 7
journeys on one side and 85,906 on the other). Every aggregate elsewhere still uses all journeys.

## Why a precompute step

The raw sample is ~410 MB over 6 CSVs and the run writes ~1.4 GB
(`*_events_canonical.csv` alone is 440 MB). `prepare_dashboard_data.py` reduces all of it
to a ~11 MB cache in `output/dashboard_cache/`:

- `eda.json` — per-file raw statistics: volumes, session shape, gap distributions,
  vocabulary/taxonomy pressure signals, cross-platform vocabulary overlap, one real
  raw session per file.
- `train_run.json` — run config, fit metrics, journey-level distributions computed by
  streaming `*_journeys.csv`, and holdout scoring summaries.
- `inference_{platform}.parquet` — the scored T5 journeys, slimmed to the columns the
  business page uses.
- `showcase.json` — one real session chosen deterministically (two journeys, two different
  journey types from two different business families, one flagged, one with a predicted next
  action), holding its raw event rows plus the journeys the model produced from them. This is
  what makes the overview page's input→output panel traceable rather than illustrative.

Rebuild the cache after every new training run.

## Known data caveat

The scored files in `output/test/` carry a broken `cluster_name_en` /
`business_family_code` pair — every row inherits the noise cluster's label. The precompute
step ignores those two columns and re-joins names from the run's `cluster_name_mapping.csv`
on `(platform, cluster_id)`, which is the authoritative source. If that bug is fixed upstream,
the join can be dropped.
