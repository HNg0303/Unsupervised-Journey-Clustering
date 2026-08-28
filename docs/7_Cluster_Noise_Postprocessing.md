# B/C hierarchical cluster post-processing

This step keeps the strict C run as the primary catalog and uses the lenient B
run to recover useful information from C's HDBSCAN noise.  It never overwrites
the original fitted labels.

```powershell
python scripts/postprocess_cluster_runs.py `
  --b-run output/journey_runs/<B-run> `
  --c-run output/journey_runs/<C-run> `
  --name L2_ng1-3_C-mcs100-ms5_B-mcs50-ms3_postprocessed
```

Assignment precedence:

1. `C_primary`: C assigned the journey, so C remains authoritative.
2. `B_existing_secondary`: C returned noise but B fitted the journey into a
   cluster.  The B label becomes a secondary archetype.
3. `B_borderline_secondary`: both runs returned noise, but the journey passes
   B cluster-specific centroid-distance and Markov-likelihood thresholds and a
   nearest-versus-second-nearest distance margin.
4. Remaining noise is classified as `unassigned_recurring`,
   `unassigned_friction`, or `unassigned_novel`.

Cluster IDs are namespaced as `C:<id>` and `B:<id>` because HDBSCAN integer IDs
from different fits are not directly comparable.

Per platform, the step writes:

- `<platform>_journeys_postprocessed.csv`: every C journey plus B evidence,
  assignment tier, distances, likelihoods, recurrence, and friction flags.
- `<platform>_secondary_cluster_catalog.{csv,json}`: B medoids and metadata for
  secondary archetypes actually used by C-noise journeys.
- `<platform>_unresolved_noise.csv`: recurring, friction-like, and novel
  journeys that remain unassigned.
- `<platform>_assignment_summary.csv` and
  `<platform>_postprocess_summary.json`: coverage and assignment counts.
- `postprocess_config.json` and `RUN_REPORT.md`: provenance and combined report.

The default soft-assignment gates are B cluster distance p95, B cluster Markov
p05, and a 10% distance margin.  They can be changed with
`--distance-quantile`, `--markov-quantile`, and `--min-distance-margin`.

The historical B/C naming route is no longer part of the current production pipeline.
For the current partitioned run, apply the manually reviewed mapping to the scored
output:

```powershell
python scripts/apply_mapping_name.py `
  --input output/scores/<bundle>/android/model_version=<version>/platform=android `
  --platform android `
  --mapping output/scores/<bundle>/Cluster_naming.csv `
  --output output/scores/<bundle>/android_all_named.csv
```

Names use the medoid path and top three ranked n-grams.  A sole action token
does not claim a completed business journey unless a matching destination view
also appears in the medoid or displayed n-grams; in that case the naming logic
falls back to the coarser medoid surface.

Run hierarchical inference:

```powershell
python scripts/score_partitioned_events.py `
  --platform android `
  --input data/test_data/test_data_android.csv `
  --postprocess-run output/journey_runs/L2_ng1-3_C-mcs100-ms5_B-mcs50-ms3_postprocessed `
  --output output/test/android_hierarchical_scored.csv
```

Inference prepares events once, assigns C with cluster-specific distance p95,
and evaluates B only for C-noise journeys.  A B fallback must pass its own
cluster distance p95, Markov p05, and nearest/second-nearest margin.  Output
IDs remain namespaced as `C:<id>`, `B:<id>`, or `UNKNOWN` and include the
generated cluster name and business family.
