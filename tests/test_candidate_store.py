from pathlib import Path

from service.candidate_store import CandidateStore


def test_candidate_store_round_trips_items(tmp_path: Path):
    store = CandidateStore(tmp_path / "mapping_candidates.json")
    payload = {
        "candidate_id": "c1",
        "status": "pending",
        "graph_candidate": {"specific_domain": "后处理痕迹域"},
        "mapping_candidate": {
            "detector": "FFTDetector",
            "feature": "patch_inconsistency",
        },
    }

    store.append_items([payload])
    items = store.list_items()

    assert len(items) == 1
    assert items[0]["candidate_id"] == "c1"


def test_candidate_store_updates_existing_item(tmp_path: Path):
    store = CandidateStore(tmp_path / "mapping_candidates.json")
    store.append_items(
        [
            {
                "candidate_id": "c1",
                "status": "pending",
                "mapping_candidate": {
                    "detector": "FFTDetector",
                    "feature": "patch_inconsistency",
                },
            }
        ]
    )

    updated = store.update_item("c1", {"status": "approved"})

    assert updated["status"] == "approved"
    assert store.get_item("c1")["status"] == "approved"
