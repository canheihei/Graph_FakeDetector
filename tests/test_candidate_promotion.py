import json
from pathlib import Path

from service.candidate_benchmark import promote_candidate_rules


def test_promote_candidate_rules_rewrites_mapping_file_for_selected_candidate(tmp_path: Path):
    mapping_path = tmp_path / "mapping_config.json"
    mapping_path.write_text(
        json.dumps(
            {
                "version": "1.4",
                "rules": [
                    {
                        "detector": "FFTDetector",
                        "feature": "patch_inconsistency",
                        "subdomain_id": "old-id",
                        "subdomain_label": "旧节点",
                    }
                ],
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    selected = [
        {
            "candidate_id": "c1",
            "mapping_candidate": {
                "detector": "FFTDetector",
                "feature": "patch_inconsistency",
                "subdomain_id": "new-id",
                "subdomain_label": "新节点",
            },
        }
    ]

    promote_candidate_rules(mapping_path, selected)
    updated = json.loads(mapping_path.read_text(encoding="utf-8"))

    assert updated["rules"][0]["subdomain_id"] == "new-id"
    assert updated["rules"][0]["subdomain_label"] == "新节点"
