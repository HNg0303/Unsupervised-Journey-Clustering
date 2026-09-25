# Archived B/C hierarchical post-processing

The former two-run B/C post-processing workflow is not part of the current
production pipeline. Its CLI entry point and hierarchical scoring option were
removed to keep one authoritative training and inference path.

Current production flow:

1. Partition events with `scripts/partition_raw_events.py`.
2. Prepare journeys and train with `scripts/run_partitioned_journey_pipeline.py`.
3. Score with `scripts/score_partitioned_events.py --model-run ...`.
4. Name clusters and optionally decorate score CSVs with
   `scripts/taxonomy_cluster_naming_pipeline.py`.

The reusable historical algorithms remain under
`journey_clustering.cluster_postprocess` and
`journey_clustering.hierarchical_score` for reference, but no core command
imports or deploys them.
