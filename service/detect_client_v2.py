from __future__ import annotations

from typing import Dict, List

import numpy as np


DOMAIN_WEIGHT: Dict[str, float] = {}


def decide_v2(evidence: list, mode: str = "balanced") -> dict:
    if not evidence:
        return {
            "label": "REAL",
            "confidence": 0.0,
            "confidence_interval": [0.0, 0.0],
            "domain_scores": {},
            "uncertainty": 0.0,
            "threshold": 0.5,
            "mode": mode,
        }

    domain_evidence: Dict[str, List[dict]] = {}
    for item in evidence:
        main_domain = item.get("main_domain", {}).get("name", "UnknownMainDomain")
        domain_evidence.setdefault(main_domain, []).append(item)

    print(f"[RELOAD] deciding across {len(domain_evidence)} main domains")
    domain_scores: Dict[str, float] = {}
    for domain, items in domain_evidence.items():
        confidences = [float(entry.get("confidence", 0.0)) for entry in items]
        max_confidence = max(confidences)
        avg_confidence = float(sum(confidences) / len(confidences))
        domain_scores[domain] = 0.7 * max_confidence + 0.3 * avg_confidence

    total_weighted_score = 0.0
    total_weight = 0.0
    for domain, score in domain_scores.items():
        weight = DOMAIN_WEIGHT.get(domain, 1.0)
        total_weighted_score += score * weight
        total_weight += weight

    confidence = total_weighted_score / total_weight if total_weight else 0.0
    uncertainty = float(np.std(np.array(list(domain_scores.values()), dtype=np.float32))) if len(domain_scores) > 1 else 0.0
    confidence_lower = max(0.0, confidence - uncertainty)
    confidence_upper = min(1.0, confidence + uncertainty)

    thresholds = {
        "strict": 0.7,
        "balanced": 0.5,
        "sensitive": 0.3,
    }
    threshold = thresholds.get(mode, 0.5)
    label = "FAKE" if confidence >= threshold else "REAL"

    return {
        "label": label,
        "confidence": round(confidence, 3),
        "confidence_interval": [round(confidence_lower, 3), round(confidence_upper, 3)],
        "domain_scores": {key: round(value, 3) for key, value in domain_scores.items()},
        "uncertainty": round(uncertainty, 3),
        "threshold": threshold,
        "mode": mode,
    }


def decide_v2_with_correlation(evidence: list, mode: str = "balanced") -> dict:
    return decide_v2(evidence, mode)


def decide(evidence: list) -> dict:
    result = decide_v2(evidence, mode="balanced")
    return {
        "label": result["label"],
        "confidence": result["confidence"],
    }
