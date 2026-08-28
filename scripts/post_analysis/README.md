# Independent post-analysis jobs

All jobs start from the final `*_all_named.csv` anchors produced by
`scripts/apply_mapping_name.py`. They use DuckDB so the multi-GB inference exports
are not materialised in pandas.

- `analyze_customer_metrics.py`: customer footprint, monthly activity and journey mix.
- `analyze_focus_journeys.py`: focus-object records and customer summaries for VNeID,
  support, payment, device management, notifications and Loyalty.
- `analyze_post_vneid_econtract.py`: post-signature customer behaviour around VNeID/e-contract.
- `analyze_loyalty.py`: Loyalty cohort and post-Loyalty business actions.
- `run_all.py`: optional thin orchestrator; it does not contain the analysis logic.

Each job can be rerun independently and writes its own CSV outputs under the selected
`shareholder_analysis` directory. These outputs are the source for the customer-based
dashboard page; inference volume and cluster KPIs come from
`html_dashboard_summary/` instead.
