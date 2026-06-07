from scripts.training.train_domain_generalized_calibrated_vision_detector import (
    choose_best_epoch_payload,
    summarize_epoch_selection,
)


def test_choose_best_epoch_payload_prefers_target_domains_under_guardrails():
    payloads = [
        {
            "epoch": 1,
            "mean_dataset_balanced_accuracy": 0.94,
            "val_metrics": {"balanced_accuracy": 0.94, "accuracy": 0.94},
            "per_dataset": {
                "Test": {"balanced_accuracy": 0.91},
                "Celeb-DF": {"balanced_accuracy": 0.90},
                "DFDC_Curated": {"balanced_accuracy": 0.89},
                "WildDeepfake_Curated": {"balanced_accuracy": 0.90},
            },
        },
        {
            "epoch": 2,
            "mean_dataset_balanced_accuracy": 0.93,
            "val_metrics": {"balanced_accuracy": 0.93, "accuracy": 0.93},
            "per_dataset": {
                "Test": {"balanced_accuracy": 0.92},
                "Celeb-DF": {"balanced_accuracy": 0.91},
                "DFDC_Curated": {"balanced_accuracy": 0.95},
                "WildDeepfake_Curated": {"balanced_accuracy": 0.96},
            },
        },
    ]

    best = choose_best_epoch_payload(
        payloads,
        guardrail_domains=("Test", "Celeb-DF"),
        guardrail_min_balanced_accuracy=0.90,
        target_domains=("DFDC_Curated", "WildDeepfake_Curated"),
    )

    assert best["epoch"] == 2


def test_choose_best_epoch_payload_falls_back_to_smallest_guardrail_gap():
    payloads = [
        {
            "epoch": 3,
            "mean_dataset_balanced_accuracy": 0.95,
            "val_metrics": {"balanced_accuracy": 0.95, "accuracy": 0.95},
            "per_dataset": {
                "Test": {"balanced_accuracy": 0.89},
                "Celeb-DF": {"balanced_accuracy": 0.88},
                "DFDC_Curated": {"balanced_accuracy": 0.97},
                "WildDeepfake_Curated": {"balanced_accuracy": 0.97},
            },
        },
        {
            "epoch": 4,
            "mean_dataset_balanced_accuracy": 0.93,
            "val_metrics": {"balanced_accuracy": 0.93, "accuracy": 0.93},
            "per_dataset": {
                "Test": {"balanced_accuracy": 0.895},
                "Celeb-DF": {"balanced_accuracy": 0.895},
                "DFDC_Curated": {"balanced_accuracy": 0.95},
                "WildDeepfake_Curated": {"balanced_accuracy": 0.95},
            },
        },
    ]

    best = choose_best_epoch_payload(
        payloads,
        guardrail_domains=("Test", "Celeb-DF"),
        guardrail_min_balanced_accuracy=0.90,
        target_domains=("DFDC_Curated", "WildDeepfake_Curated"),
    )

    assert best["epoch"] == 4


def test_summarize_epoch_selection_contains_guardrail_and_target_scores():
    payload = {
        "epoch": 7,
        "mean_dataset_balanced_accuracy": 0.94,
        "per_dataset": {
            "Test": {"balanced_accuracy": 0.92},
            "Celeb-DF": {"balanced_accuracy": 0.91},
            "DFDC_Curated": {"balanced_accuracy": 0.95},
            "WildDeepfake_Curated": {"balanced_accuracy": 0.96},
        },
    }

    summary = summarize_epoch_selection(
        payload,
        guardrail_domains=("Test", "Celeb-DF"),
        target_domains=("DFDC_Curated", "WildDeepfake_Curated"),
        guardrail_min_balanced_accuracy=0.90,
    )

    assert summary["guardrail_average"] == 0.915
    assert summary["target_average"] == 0.955
    assert summary["guardrail_domains"] == ["Test", "Celeb-DF"]
    assert summary["target_domains"] == ["DFDC_Curated", "WildDeepfake_Curated"]
