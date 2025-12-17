from service.neo_client import neo4j_client


# def build_evidence(activated_subdomains: list) -> list:
#     sub_ids = [x["subdomain_id"] for x in activated_subdomains]
#
#     cypher = """
#     MATCH (s:SubDomain)-[:SPECIFIC_OF]->(d:SpecificDomain)-[:KINDS_OF]->(m:MainDomain)
#     WHERE s.id IN $sub_ids
#     RETURN s.id AS sid, s.name AS sname,
#            d.name AS dname, m.name AS mname
#     """
#
#     records = neo4j_client.query(cypher, {"sub_ids": sub_ids})
#
#     evidence = []
#     for r in records:
#         score = next(x["score"] for x in activated_subdomains if x["subdomain_id"] == r["sid"])
#         evidence.append({
#             "subdomain_id": r["sid"],
#             "subdomain": r["sname"],
#             "domain": r["dname"],
#             "main_domain": r["mname"],
#             "score": score
#         })
#
#     return evidence

def build_evidence(activated_subdomains):
    return [
        {
            "subdomain": x["subdomain_id"],
            "domain": "StubDomain",
            "main_domain": "域泛化",
            "score": x["score"]
        }
        for x in activated_subdomains
    ]


DOMAIN_WEIGHT = {
    "后处理痕迹域": 1.2,
    "外观扰动域": 1.25,
    "身份属性偏移域": 1.15,
    "质量与分辨率域": 1.1
}

def decide(evidence: list) -> dict:
    total = 0.0

    for e in evidence:
        weight = DOMAIN_WEIGHT.get(e["domain"], 1.0)
        total += e["score"] * weight

    confidence = min(total / 3.0, 1.0)

    return {
        "label": "FAKE" if confidence >= 0.5 else "REAL",
        "confidence": round(confidence, 3)
    }


from config import FEATURE_THRESHOLD

FEATURE_SUBDOMAIN_MAP = {
    "jpeg_blockiness": "S08",
    "lighting_conflict": "S18",
    "pose_extreme": "S15",
    "age_texture_mismatch": "S14",
    "super_resolution_artifact": "S19"
}

def map_features(features: dict) -> list:
    activated = []

    for group in features.values():
        for name, score in group.items():
            if score >= FEATURE_THRESHOLD and name in FEATURE_SUBDOMAIN_MAP:
                activated.append({
                    "subdomain_id": FEATURE_SUBDOMAIN_MAP[name],
                    "feature": name,
                    "score": score
                })

    return activated

def extract_features(image_bytes: bytes) -> dict:
    """
    模拟图像特征提取（可替换为真实模型）
    """

    # 假设我们“检测”到了这些异常
    return {
        "frequency": {
            "jpeg_blockiness": 0.81,
            "high_freq_repeat": 0.67
        },
        "appearance": {
            "lighting_conflict": 0.79,
            "pose_extreme": 0.65
        },
        "identity": {
            "age_texture_mismatch": 0.82
        },
        "quality": {
            "super_resolution_artifact": 0.77
        }
    }
