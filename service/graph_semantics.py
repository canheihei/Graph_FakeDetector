from __future__ import annotations

import hashlib
import re
import uuid
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Set

from service.llm_chain import semantic_match


GENERIC_TERMS = {
    "异常",
    "伪影",
    "痕迹",
    "伪造痕迹",
    "图像异常",
    "图像模糊",
    "模糊",
    "质量差",
    "低质量",
    "噪声",
    "不自然",
    "异常区域",
    "未知特征",
}

FEATURE_TEMPLATES: Dict[str, Dict[str, str]] = {
    "high_freq_energy": {
        "display_name": "高频残差尖峰",
        "canonical_name": "high_frequency_residual_spike",
        "description": "频域高频能量异常抬升，通常对应生成纹理过锐、重采样残差聚集或局部合成边界泄漏。",
    },
    "lighting_conflict": {
        "display_name": "光照一致性冲突",
        "canonical_name": "illumination_consistency_conflict",
        "description": "面部不同区域的受光方向、阴影衰减或高光分布不一致，常见于伪造融合或生成补全失败。",
    },
    "pose_extreme": {
        "display_name": "极端姿态结构失真",
        "canonical_name": "extreme_pose_structural_distortion",
        "description": "大角度侧脸、俯仰头等极端姿态下，五官拓扑与轮廓结构连续性被破坏。",
    },
    "boundary_inconsistency": {
        "display_name": "边界融合不连续",
        "canonical_name": "boundary_blending_discontinuity",
        "description": "人脸与头发、皮肤、背景交界处出现融合过渡突变，表现为边缘纹理和颜色场不连续。",
    },
    "weighted_ensemble_score": {
        "display_name": "跨检测器共识异常",
        "canonical_name": "cross_detector_consensus_anomaly",
        "description": "多个检测器在同一图像上形成一致异常判断，说明伪造痕迹具有跨视角稳定性。",
    },
    "max_anomaly_score": {
        "display_name": "峰值异常响应",
        "canonical_name": "peak_anomaly_response",
        "description": "局部异常区域存在高强度响应，通常对应合成缺陷最集中的结构区域。",
    },
    "anomaly_coverage": {
        "display_name": "异常区域覆盖扩张",
        "canonical_name": "anomaly_coverage_expansion",
        "description": "异常不再局限于局部点状区域，而是扩展到更大的人脸结构范围。",
    },
    "detector_agreement": {
        "display_name": "检测器判别一致性",
        "canonical_name": "detector_decision_agreement",
        "description": "多检测器输出的判别方向高度一致，可作为跨域稳健证据的辅助信号。",
    },
    "fake_probability": {
        "display_name": "全局伪造概率异常",
        "canonical_name": "global_fake_probability_anomaly",
        "description": "主干模型在全局视觉表征上识别到稳定的伪造偏差，是对人脸生成异常的综合响应。",
    },
    "vit_fake_prob": {
        "display_name": "视觉变换器伪造响应",
        "canonical_name": "vit_fake_response_pattern",
        "description": "Transformer 视觉表征对非自然伪造纹理与空间依赖产生显著响应。",
    },
    "vit_prediction_entropy": {
        "display_name": "判别不确定性抬升",
        "canonical_name": "prediction_uncertainty_elevation",
        "description": "模型决策熵升高，说明样本存在边界型伪造特征或跨域分布漂移。",
    },
}

SEMANTIC_ALIAS_REPLACEMENTS = {
    "身份语义": "身份属性",
    "身份边界": "身份属性",
    "年龄一致性": "年龄属性",
    "生物属性": "生理属性",
    "种族特征": "族裔属性",
    "性别二态性": "性别属性",
    "不匹配": "冲突",
    "错配": "冲突",
    "混叠": "冲突",
    "失配": "冲突",
}

ONTOLOGY_ANCHOR_GROUPS: Dict[str, tuple[str, ...]] = {
    "identity": ("身份", "identity"),
    "age": ("年龄", "age", "幼态", "老化"),
    "gender": ("性别", "gender", "male", "female"),
    "ethnicity": ("种族", "族裔", "ethnic", "race"),
    "physiology": ("生理", "生物", "biological"),
    "expression": ("表情", "expression"),
    "pose": ("姿态", "姿势", "pose"),
    "lighting": ("光照", "照明", "illumination", "shadow"),
    "boundary": ("边界", "轮廓", "boundary"),
    "texture": ("纹理", "高频", "频域", "texture", "frequency"),
    "conflict": ("冲突", "不一致", "矛盾", "conflict"),
    "shift": ("偏移", "漂移", "shift", "drift"),
    "blur": ("模糊", "blur"),
    "uncertainty": ("不确定", "熵", "uncertainty"),
}

CATEGORY_ANCHORS = {
    "identity",
    "age",
    "gender",
    "ethnicity",
    "physiology",
    "expression",
    "pose",
    "lighting",
    "boundary",
    "texture",
}

RELATION_ANCHORS = {"conflict", "shift", "blur", "uncertainty"}

CURATED_ONTOLOGY_PROFILES: Dict[str, Dict[str, object]] = {
    "identity_attribute_shift": {
        "domain_terms": ("身份", "属性"),
        "generic_terms": {
            "身份属性偏移",
            "身份语义偏移",
            "身份偏移",
            "身份异常",
            "身份属性异常",
        },
        "prototypes": [
            {
                "display_name": "年龄属性偏移",
                "canonical_name": "identity_age_attribute_shift",
                "description": "面部呈现的年龄线索与整体身份语义不一致，常表现为幼态化、老化程度或年龄结构被异常重写。",
                "aliases": ("年龄一致性偏移", "年龄语义偏移", "年龄错配", "年龄漂移", "年龄属性冲突"),
                "category_terms": ("年龄", "幼态", "老化"),
                "relation_terms": ("偏移", "漂移", "冲突", "异常", "偏差", "矛盾"),
            },
            {
                "display_name": "性别属性冲突",
                "canonical_name": "identity_gender_attribute_conflict",
                "description": "面部的性别表达与身份属性之间出现冲突，通常体现在性别二态特征被混合或翻转。",
                "aliases": ("性别二态性冲突", "性别错配", "性别偏移", "性别属性混淆", "性别表征矛盾"),
                "category_terms": ("性别", "男", "女"),
                "relation_terms": ("冲突", "错配", "偏移", "异常", "混淆", "矛盾"),
            },
            {
                "display_name": "族裔特征混叠",
                "canonical_name": "identity_ethnicity_feature_conflict",
                "description": "面部的族裔线索被跨身份混合，导致肤色、骨相或局部特征组合出现不自然的族裔混叠。",
                "aliases": ("种族特征混叠", "种族错配", "族裔错配", "族裔偏移", "种族特征偏差"),
                "category_terms": ("种族", "族裔"),
                "relation_terms": ("冲突", "混叠", "错配", "偏移", "偏差", "混淆"),
            },
            {
                "display_name": "生理属性错配",
                "canonical_name": "identity_physiology_attribute_conflict",
                "description": "面部生理属性与身份表征之间不一致，常见于器官比例、成熟度或人体属性被异常组合。",
                "aliases": ("生物属性错配", "生理错配", "生理属性偏移"),
                "category_terms": ("生物", "生理"),
                "relation_terms": ("冲突", "错配", "偏移", "异常"),
            },
            {
                "display_name": "身份边界模糊",
                "canonical_name": "identity_boundary_ambiguity",
                "description": "面部身份边界缺乏清晰一致的语义约束，导致身份主体特征呈现模糊、漂移或双重归属。",
                "aliases": ("身份边界模糊", "主体边界模糊"),
                "category_terms": ("身份", "主体"),
                "relation_terms": ("边界", "模糊"),
            },
        ],
    }
}


def _collapse_spaces(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def _normalize_label(text: str) -> str:
    text = str(text or "").strip()
    text = text.replace("_", " ").replace("-", " ")
    text = re.sub(r"[\"'`]", "", text)
    text = re.sub(r"[，,；;:：]+", " ", text)
    return _collapse_spaces(text)


def _normalize_description(text: str) -> str:
    text = _collapse_spaces(str(text or ""))
    if not text:
        return ""
    if text[-1] not in {"。", ".", "!", "！"}:
        text = f"{text}。"
    return text


def _apply_semantic_aliases(text: str) -> str:
    normalized = _normalize_label(text)
    for source, target in SEMANTIC_ALIAS_REPLACEMENTS.items():
        normalized = normalized.replace(source, target)
    return _collapse_spaces(normalized)


def _dedup_key(text: str) -> str:
    normalized = _normalize_label(text).lower()
    normalized = normalized.replace(" ", "")
    return normalized


def _canonical_from_text(text: str, fallback_prefix: str) -> str:
    ascii_text = re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")
    if ascii_text:
        return ascii_text
    digest = hashlib.sha1(text.encode("utf-8")).hexdigest()[:10]
    return f"{fallback_prefix}_{digest}"


def _build_semantic_text(item: Dict[str, str]) -> str:
    parts = [
        item.get("display_name") or item.get("name") or "",
        item.get("name") or "",
        item.get("canonical_name") or "",
        item.get("describe") or "",
    ]
    return _collapse_spaces(" ".join(part for part in parts if part))


def _build_ontology_signature(item: Dict[str, str]) -> Set[str]:
    text = _apply_semantic_aliases(_build_semantic_text(item))
    lowered = text.lower()
    signature: Set[str] = set()
    for anchor, terms in ONTOLOGY_ANCHOR_GROUPS.items():
        if any(term in text or term in lowered for term in terms):
            signature.add(anchor)
    return signature


def _merge_existing_subdomain(candidate: Dict[str, str], existing: Dict[str, str]) -> Dict[str, str]:
    merged = dict(candidate)
    merged["name"] = existing.get("name") or candidate["name"]
    merged["display_name"] = (
        existing.get("display_name")
        or existing.get("name")
        or candidate.get("display_name")
        or candidate["name"]
    )
    merged["canonical_name"] = (
        existing.get("canonical_name")
        or candidate.get("canonical_name")
        or _canonical_from_text(merged["name"], "subdomain")
    )
    merged["describe"] = existing.get("describe") or candidate["describe"]
    merged["sub_id"] = existing.get("sub_id") or candidate["sub_id"]
    return merged


def _resolve_ontology_profile(specific_domain_name: str) -> Optional[Dict[str, object]]:
    normalized = _normalize_label(specific_domain_name)
    for profile in CURATED_ONTOLOGY_PROFILES.values():
        domain_terms = tuple(profile.get("domain_terms", ()))
        if domain_terms and all(term in normalized for term in domain_terms):
            return profile
    return None


def _match_curated_prototype(
    *,
    label: str,
    describe: str,
    specific_domain_name: str,
) -> Optional[Dict[str, str]]:
    profile = _resolve_ontology_profile(specific_domain_name)
    if profile is None:
        return None

    search_text = _apply_semantic_aliases(f"{label} {describe}")
    for prototype in profile.get("prototypes", []):
        aliases = tuple(prototype.get("aliases", ()))
        if any(alias and alias in search_text for alias in aliases):
            return {
                "name": prototype["display_name"],
                "display_name": prototype["display_name"],
                "canonical_name": prototype["canonical_name"],
                "describe": prototype["description"],
            }

        category_terms = tuple(prototype.get("category_terms", ()))
        relation_terms = tuple(prototype.get("relation_terms", ()))
        has_category = any(term in search_text for term in category_terms)
        has_relation = any(term in search_text for term in relation_terms)
        if has_category and has_relation:
            return {
                "name": prototype["display_name"],
                "display_name": prototype["display_name"],
                "canonical_name": prototype["canonical_name"],
                "describe": prototype["description"],
            }

    generic_terms = {str(item) for item in profile.get("generic_terms", set())}
    if label in generic_terms:
        return {"skip": "1"}

    return None


def match_existing_subdomain(
    candidate: Dict[str, str],
    existing_subdomains: Iterable[Dict[str, str]],
    semantic_threshold: float,
) -> Optional[Dict[str, str]]:
    candidate_label = _normalize_label(candidate.get("display_name") or candidate.get("name"))
    if not candidate_label:
        return None

    candidate_canonical = str(candidate.get("canonical_name") or "").strip().lower()
    candidate_key = _dedup_key(candidate_label)
    candidate_signature = _build_ontology_signature(candidate)
    best_anchor_match: Optional[Dict[str, str]] = None
    best_anchor_score = 0.0
    existing_records = list(existing_subdomains)

    for existing in existing_records:
        existing_name = _normalize_label(existing.get("name"))
        existing_display = _normalize_label(existing.get("display_name") or existing_name)
        existing_canonical = str(existing.get("canonical_name") or "").strip().lower()

        if candidate_canonical and existing_canonical and candidate_canonical == existing_canonical:
            return existing

        if candidate_key in {
            _dedup_key(existing_name),
            _dedup_key(existing_display),
        }:
            return existing

        existing_signature = _build_ontology_signature(existing)
        category_overlap = len((candidate_signature & CATEGORY_ANCHORS) & existing_signature)
        relation_overlap = len((candidate_signature & RELATION_ANCHORS) & existing_signature)
        anchor_overlap = len(candidate_signature & existing_signature)
        if category_overlap >= 1 and relation_overlap >= 1 and anchor_overlap >= 2:
            score = 0.90 + 0.02 * min(anchor_overlap, 3)
            if score > best_anchor_score:
                best_anchor_score = score
                best_anchor_match = existing

    if best_anchor_match is not None and best_anchor_score >= max(semantic_threshold, 0.90):
        return best_anchor_match

    candidate_category_signature = candidate_signature & CATEGORY_ANCHORS
    fallback_records = existing_records
    if candidate_category_signature:
        fallback_records = [
            existing
            for existing in existing_records
            if _build_ontology_signature(existing) & candidate_category_signature
        ]
        if not fallback_records:
            return None

    label_candidates: List[str] = []
    label_to_record: Dict[str, Dict[str, str]] = {}
    semantic_candidates: List[str] = []
    semantic_to_record: Dict[str, Dict[str, str]] = {}
    for existing in fallback_records:
        label = existing.get("display_name") or existing.get("name") or ""
        if label and label not in label_to_record:
            label_to_record[label] = existing
            label_candidates.append(label)

        semantic_text = _build_semantic_text(existing)
        if semantic_text and semantic_text not in semantic_to_record:
            semantic_to_record[semantic_text] = existing
            semantic_candidates.append(semantic_text)

    if label_candidates:
        matched_label = semantic_match(
            candidate_label,
            label_candidates,
            max(semantic_threshold, 0.92),
        )
        if matched_label != candidate_label:
            return label_to_record[matched_label]

    candidate_semantic_text = _build_semantic_text(candidate)
    if semantic_candidates and candidate_semantic_text:
        matched_text = semantic_match(
            candidate_semantic_text,
            semantic_candidates,
            max(semantic_threshold + 0.04, 0.90),
        )
        if matched_text != candidate_semantic_text:
            return semantic_to_record[matched_text]

    return None


@dataclass(frozen=True)
class ResolvedSpecificDomain:
    id: str
    name: str
    describe: str
    main_domain: str
    main_describe: str


class GraphSemanticGovernance:
    def __init__(self, neo4j_client):
        self._neo4j_client = neo4j_client

    def _resolve_default_main_domain(self) -> tuple[str, str]:
        main_domains = self._neo4j_client.list_main_domains()
        if not main_domains:
            return "未分类主域", ""

        for domain in main_domains:
            if str(domain.get("name", "") or "").strip() == "域泛化":
                return domain["name"], str(domain.get("describe", "") or "")

        non_placeholder_domains = [
            domain
            for domain in main_domains
            if str(domain.get("name", "") or "").strip() not in {"", "未分类主域", "未连接主域"}
        ]
        if len(non_placeholder_domains) == 1:
            domain = non_placeholder_domains[0]
            return domain["name"], str(domain.get("describe", "") or "")

        if len(main_domains) == 1:
            domain = main_domains[0]
            return domain["name"], str(domain.get("describe", "") or "")

        return "未分类主域", ""

    def resolve_specific_domain(
        self,
        candidate_name: str,
        fallback_domain: Optional[Dict] = None,
        threshold: float = 0.78,
    ) -> Optional[ResolvedSpecificDomain]:
        domains = self._neo4j_client.list_specific_domains(include_main_domain=True)
        if not domains and fallback_domain is None:
            return None

        if fallback_domain:
            matched = self._match_domain(candidate_name, domains, threshold) if domains else fallback_domain
            if matched is None:
                matched = fallback_domain
        else:
            matched = self._match_domain(candidate_name, domains, threshold)

        if matched is None:
            if not domains:
                return None
            matched = domains[0]

        default_main_name, default_main_describe = self._resolve_default_main_domain()
        main_name = (
            matched.get("main_domain")
            or self._neo4j_client.get_main_domain_name_by_specific_domain(matched["name"])
            or default_main_name
        )
        main_describe = (
            matched.get("main_describe")
            or self._neo4j_client.get_main_domain_describe(main_name)
            or default_main_describe
        )
        return ResolvedSpecificDomain(
            id=matched.get("id", str(uuid.uuid4())),
            name=matched["name"],
            describe=matched.get("describe", ""),
            main_domain=main_name,
            main_describe=main_describe,
        )

    def normalize_iteration_payload(
        self,
        raw_payload: Dict,
        *,
        prompt: str,
        matched_domain: str,
        semantic_threshold: float,
    ) -> Dict:
        resolved = self.resolve_specific_domain(
            raw_payload.get("specific_domain", matched_domain),
            fallback_domain={"name": matched_domain},
            threshold=max(semantic_threshold, 0.78),
        )
        if resolved is None:
            raise ValueError("No specific domain available for graph evolution")

        raw_subdomains = raw_payload.get("subdomain", [])
        normalized_subdomains = self._normalize_subdomain_batch(
            raw_subdomains,
            specific_domain_name=resolved.name,
            semantic_threshold=max(semantic_threshold, 0.86),
            fallback_prefix=_canonical_from_text(resolved.name, "specific_domain"),
            existing_subdomains=self._neo4j_client.list_subdomain_records(resolved.name),
        )
        return {
            "main_domain": resolved.main_domain,
            "main_describe": resolved.main_describe,
            "specific_domain": resolved.name,
            "describe": _normalize_description(raw_payload.get("describe") or resolved.describe),
            "specific_id": resolved.id,
            "subdomain": normalized_subdomains,
            "semantic_source": "iterate",
            "semantic_prompt": prompt,
            "semantic_version": "graph_semantics_v2",
        }

    def build_feature_semantic_draft(
        self,
        *,
        detector: str,
        feature: str,
        score: float,
        specific_domain: Dict,
    ) -> Dict[str, str]:
        template = FEATURE_TEMPLATES.get(feature)
        if template:
            description = template["description"]
            if specific_domain.get("name"):
                description = f"{description} 该节点归属于“{specific_domain['name']}”语义下的稳定伪造证据。"
            return {
                "name": template["display_name"],
                "canonical_name": template["canonical_name"],
                "describe": description,
            }

        feature_label = _normalize_label(feature)
        display_name = f"{feature_label}证据"
        canonical_name = _canonical_from_text(
            f"{detector}_{feature}",
            "feature_signal",
        )
        description = (
            f"由检测器 {detector} 捕获的特征“{feature}”在当前样本上达到 {score:.3f}，"
            f"被抽象为“{specific_domain.get('name', '目标域')}”下的候选稳定证据节点。"
        )
        return {
            "name": display_name,
            "canonical_name": canonical_name,
            "describe": description,
        }

    def normalize_manual_subdomain(
        self,
        *,
        name: str,
        describe: str,
        specific_domain_name: str,
        fallback_prefix: str,
    ) -> Dict[str, str]:
        normalized_name = _normalize_label(name)
        normalized_describe = _normalize_description(describe)
        if not normalized_describe and normalized_name:
            normalized_describe = (
                f"该节点表示“{specific_domain_name}”下经人工确认的稳定伪造语义证据，"
                f"名称为“{normalized_name}”。"
            )
        return {
            "name": normalized_name,
            "display_name": normalized_name,
            "canonical_name": _canonical_from_text(normalized_name, fallback_prefix),
            "describe": normalized_describe,
            "sub_id": str(uuid.uuid4()),
        }

    def _normalize_subdomain_batch(
        self,
        subdomains: Iterable[Dict],
        *,
        specific_domain_name: str,
        semantic_threshold: float,
        fallback_prefix: str,
        existing_subdomains: Optional[Iterable[Dict]] = None,
    ) -> List[Dict]:
        normalized: List[Dict] = []
        exact_seen: Dict[str, int] = {}
        existing_records = list(existing_subdomains or [])
        strict_profile = _resolve_ontology_profile(specific_domain_name)

        for item in subdomains:
            label = _normalize_label(item.get("name"))
            if not label or label in GENERIC_TERMS:
                continue

            curated_match = _match_curated_prototype(
                label=label,
                describe=_normalize_description(item.get("describe")),
                specific_domain_name=specific_domain_name,
            )
            if curated_match and curated_match.get("skip") == "1":
                continue
            if curated_match is None and strict_profile is not None:
                continue
            if curated_match:
                label = curated_match["name"]

            dedup_key = _dedup_key(label)
            if dedup_key in exact_seen:
                continue

            existing_labels = [candidate["name"] for candidate in normalized]
            if existing_labels:
                matched = semantic_match(label, existing_labels, semantic_threshold)
                if matched != label:
                    continue

            normalized_item = {
                "name": label,
                "display_name": label,
                "canonical_name": item.get("canonical_name")
                or _canonical_from_text(label, fallback_prefix),
                "describe": _normalize_description(item.get("describe")),
                "sub_id": item.get("sub_id") or str(uuid.uuid4()),
            }
            if curated_match:
                normalized_item["canonical_name"] = curated_match["canonical_name"]
                normalized_item["describe"] = curated_match["describe"]
            if not normalized_item["describe"]:
                normalized_item["describe"] = (
                    f"该节点表示“{specific_domain_name}”下可复用的稳定伪造语义证据，"
                    f"名称为“{label}”。"
                )

            matched_existing = match_existing_subdomain(
                normalized_item,
                existing_records,
                semantic_threshold=max(semantic_threshold, 0.90),
            )
            if matched_existing is not None:
                normalized_item = _merge_existing_subdomain(normalized_item, matched_existing)
                dedup_key = (
                    matched_existing.get("sub_id")
                    or matched_existing.get("canonical_name")
                    or _dedup_key(matched_existing.get("display_name") or matched_existing.get("name"))
                )

            if dedup_key in exact_seen:
                continue

            exact_seen[dedup_key] = 1
            normalized.append(normalized_item)

        return normalized[:5]

    @staticmethod
    def _match_domain(
        candidate_name: str,
        domains: List[Dict],
        threshold: float,
    ) -> Optional[Dict]:
        if not domains:
            return None

        candidate_name = _normalize_label(candidate_name)
        if not candidate_name:
            return None

        exact = next(
            (domain for domain in domains if domain["name"] == candidate_name),
            None,
        )
        if exact:
            return exact

        domain_names = [domain["name"] for domain in domains]
        matched_name = semantic_match(candidate_name, domain_names, threshold)
        if matched_name == candidate_name:
            return None

        return next((domain for domain in domains if domain["name"] == matched_name), None)
