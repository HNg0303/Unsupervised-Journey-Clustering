import importlib.util
from datetime import date
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("daily_journey_job", ROOT / "scripts" / "daily_journey_job.py")
job = importlib.util.module_from_spec(spec)
spec.loader.exec_module(job)


def scores_frame() -> pd.DataFrame:
    row = {
        "journey_id": "events::J000001", "session_id": "s1", "device_id": "d1", "customer_id": "6039276.0",
        "start_ts": "2026-03-25 15:48:15+00:00", "end_ts": "2026-03-25 15:48:42.5+00:00",
        "boundary_reason": "session_start", "n_events_raw": 3, "n_events_final": 2, "n_loop_removed": 0,
        "action_ratio": 0.5, "back_rate": 0.0, "revisit_ratio": 0.0, "median_gap_s": 1.0,
        "p90_gap_s": 2.0, "max_gap_s": 3.0, "entry_token": "view@HOME", "exit_token": "action@HOME#x",
        "sequence": "view@HOME -> action@HOME#x", "distance_to_centroid": 0.1, "markov_logprob": -2.0,
        "geometric_anomaly": "False", "generative_anomaly": "True", "severe_anomaly": "False",
        "friction_flags": "unknown_archetype|excessive_back", "behavioral_friction_flags": "excessive_back",
        "next_action": "view@X", "next_action_share": 0.5, "cluster": 7,
        "scored_at": "2026-10-14 07:04:43.410030+00:00",
    }
    noise = dict(row, journey_id="events::J000002", cluster=-1, customer_id=None, behavioral_friction_flags=None)
    return pd.DataFrame([row, noise])


def test_to_summary_matches_journey_summary_schema():
    names = {7: {"taxonomy_id": "M05.06.03"}, -1: {"taxonomy_id": ""}}
    out = job.to_summary(scores_frame(), platform="android", batch_date=date(2026, 3, 25),
                         model_version="678_20260929_090904", names=names)

    assert list(out.columns) == list(job.SUMMARY_COLUMNS)
    assert "next_action" not in out.columns
    first, noise = out.iloc[0], out.iloc[1]
    assert first["batch_date"] == "2026-03-25"
    assert first["journey_id"] == "2026-03-25_android::J000001"
    assert first["taxonomy_id"] == "M05.06.03"
    assert pd.isna(noise["taxonomy_id"]) and noise["cluster_id"] == -1
    assert first["sequence"] == "view@HOME,action@HOME#x"
    assert first["customer_id"] == "6039276"
    assert first["start_ts"] == "2026-03-25 15:48:15.000"
    assert first["end_ts"] == "2026-03-25 15:48:42.500"
    assert first["scored_at"] == "2026-10-14 07:04:43.410030"
    assert (first["geometric_anomaly"], first["generative_anomaly"]) == (0, 1)
    assert first["friction_flags"] == "excessive_back"
    assert pd.isna(noise["friction_flags"])


def test_dedupe_events_restores_send_order(tmp_path):
    source = tmp_path / "export.csv"
    pd.DataFrame(
        {
            "record_id": ["3-9-0", "0-5-1", "0-5-0", "0-5-0"],
            "session_id": ["s", "s", "s", "s"],
            "platform": ["Android"] * 4,
            "key": ["c", "b", "a", "a"],
            "client_time": [1000, 1000, 1000, 1000],
            "kafka_partition": [3, 0, 0, 0],
            "kafka_offset": [9, 5, 5, 5],
            "kafka_timestamp": [1050, 1020, 1020, 1020],
        }
    ).to_csv(source, index=False)
    target = tmp_path / "events.csv"

    stats = job.dedupe_events(source, target)

    assert pd.read_csv(target)["key"].tolist() == ["a", "b", "c"]
    assert stats["duplicates_dropped"] == 1
    assert not source.exists()
