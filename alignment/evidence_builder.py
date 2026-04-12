"""Evidence builder for aligned subdomains."""

from __future__ import annotations

import logging
from typing import Any, Dict, List

from alignment.models import ActivatedSubDomain
from service.neo_client import neo4j_client


class EvidenceBuilder:
    QUERY_BY_IDS = """
    MATCH (s:SubDomain)
    WHERE s.sub_id IN $sub_ids
    MATCH (s)-[:SPECIFIC_OF]->(d:SpecificDomain)
    OPTIONAL MATCH (d)-[:KINDS_OF]->(m:MainDomain)
    RETURN s.sub_id AS sub_id,
           s.name AS raw_name,
           s.display_name AS display_name,
           s.canonical_name AS canonical_name,
           coalesce(s.display_name, s.name, s.canonical_name) AS sub_name,
           d.name AS specific_name,
           coalesce(m.name, '未连接主域') AS main_name
    """

    QUERY_BY_LABELS = """
    MATCH (s:SubDomain)-[:SPECIFIC_OF]->(d:SpecificDomain)
    OPTIONAL MATCH (d)-[:KINDS_OF]->(m:MainDomain)
    WHERE coalesce(s.display_name, '') IN $labels
       OR coalesce(s.name, '') IN $labels
       OR coalesce(s.canonical_name, '') IN $labels
    RETURN s.sub_id AS sub_id,
           s.name AS raw_name,
           s.display_name AS display_name,
           s.canonical_name AS canonical_name,
           coalesce(s.display_name, s.name, s.canonical_name) AS sub_name,
           d.name AS specific_name,
           coalesce(m.name, '未连接主域') AS main_name
    """

    def __init__(self) -> None:
        self._logger = logging.getLogger(__name__)
        self._graph_query_failed = False
        self._graph_fallback_failed = False
        self._last_diagnostics: Dict[str, Any] = self.empty_diagnostics()

    @staticmethod
    def _normalize_lookup_key(value: str) -> str:
        return str(value or "").strip().lower().replace("_", "").replace("-", "").replace(" ", "")

    @staticmethod
    def empty_diagnostics() -> Dict[str, Any]:
        return {
            "requested_subdomains": 0,
            "id_matched": 0,
            "label_fallback_matched": 0,
            "unresolved_subdomains": 0,
            "unresolved_rate": 0.0,
            "fallback_triggered": False,
            "graph_query_degraded": False,
            "fallback_query_degraded": False,
        }

    def get_last_diagnostics(self) -> Dict[str, Any]:
        return dict(self._last_diagnostics)

    @staticmethod
    def _record_to_graph_info(record: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "sub_id": record["sub_id"],
            "sub_name": record["sub_name"],
            "raw_name": record.get("raw_name"),
            "display_name": record.get("display_name"),
            "canonical_name": record.get("canonical_name"),
            "specific_name": record["specific_name"],
            "main_name": record["main_name"],
        }

    def _query_by_ids(self, sub_ids: List[str]) -> Dict[str, Dict[str, Any]]:
        if not sub_ids:
            return {}

        records = neo4j_client.query(self.QUERY_BY_IDS, {"sub_ids": sub_ids})
        return {
            record["sub_id"]: self._record_to_graph_info(record)
            for record in records
        }

    def _query_by_labels(self, labels: List[str]) -> Dict[str, Dict[str, Any]]:
        if not labels:
            return {}

        records = neo4j_client.query(self.QUERY_BY_LABELS, {"labels": labels})
        label_lookup: Dict[str, Dict[str, Any]] = {}
        for record in records:
            graph_info = self._record_to_graph_info(record)
            aliases = {
                graph_info["sub_name"],
                graph_info.get("raw_name"),
                graph_info.get("display_name"),
                graph_info.get("canonical_name"),
            }
            for alias in aliases:
                if not alias:
                    continue
                if alias not in label_lookup:
                    label_lookup[alias] = graph_info
                normalized = self._normalize_lookup_key(alias)
                if normalized and normalized not in label_lookup:
                    label_lookup[normalized] = graph_info
        return label_lookup

    def build_with_diagnostics(
        self,
        activated_subdomains: List[ActivatedSubDomain],
    ) -> tuple[List[Dict[str, Any]], Dict[str, Any]]:
        if not activated_subdomains:
            diagnostics = self.empty_diagnostics()
            self._last_diagnostics = diagnostics
            return [], diagnostics

        sub_ids = list(dict.fromkeys(item.subdomain_id for item in activated_subdomains))
        self._logger.debug(
            "[RELOAD] building evidence for %s activated subdomains",
            len(activated_subdomains),
        )

        try:
            graph_lookup = self._query_by_ids(sub_ids)
        except Exception as exc:
            if not self._graph_query_failed:
                self._logger.warning(
                    "[WARN] evidence builder degraded: failed to query graph by id: %s",
                    exc,
                )
                self._graph_query_failed = True
            diagnostics = self.empty_diagnostics()
            diagnostics.update(
                {
                    "requested_subdomains": len(activated_subdomains),
                    "unresolved_subdomains": len(activated_subdomains),
                    "unresolved_rate": 1.0,
                    "graph_query_degraded": True,
                }
            )
            self._last_diagnostics = diagnostics
            return [], diagnostics
        unresolved = [item for item in activated_subdomains if item.subdomain_id not in graph_lookup]

        fallback_lookup: Dict[str, Dict[str, Any]] = {}
        fallback_triggered = False
        fallback_degraded = False
        if unresolved:
            fallback_labels = sorted(
                {
                    item.subdomain_label
                    for item in unresolved
                    if item.subdomain_label
                }
            )
            try:
                fallback_lookup = self._query_by_labels(fallback_labels)
                fallback_triggered = True
                self._logger.info(
                    "[WARN] evidence lookup fallback triggered for %s subdomains",
                    len(unresolved),
                )
            except Exception as exc:
                fallback_triggered = True
                fallback_degraded = True
                if not self._graph_fallback_failed:
                    self._logger.warning(
                        "[WARN] evidence builder degraded: failed label fallback lookup: %s",
                        exc,
                    )
                    self._graph_fallback_failed = True

        unresolved_count = 0
        id_matched = 0
        fallback_matched = 0

        evidence_list: List[Dict[str, Any]] = []
        for item in activated_subdomains:
            graph_info = graph_lookup.get(item.subdomain_id)
            if graph_info is not None:
                id_matched += 1
            else:
                lookup_keys = [item.subdomain_label]
                normalized_key = self._normalize_lookup_key(item.subdomain_label)
                if normalized_key:
                    lookup_keys.append(normalized_key)
                for key in lookup_keys:
                    if key in fallback_lookup:
                        graph_info = fallback_lookup[key]
                        fallback_matched += 1
                        break
            if graph_info is None:
                graph_info = {}
            sub_name = graph_info.get("sub_name", item.subdomain_label)
            specific_name = graph_info.get("specific_name", "UnknownSpecificDomain")
            main_name = graph_info.get("main_name", "UnknownMainDomain")
            if not graph_info:
                unresolved_count += 1

            evidence_list.append(
                {
                    "sub_domain": {
                        "id": item.subdomain_id,
                        "name": sub_name,
                    },
                    "specific_domain": {
                        "name": specific_name,
                    },
                    "main_domain": {
                        "name": main_name,
                    },
                    "subdomain": sub_name,
                    "subdomain_name": sub_name,
                    "domain": specific_name,
                    "domain_name": specific_name,
                    "main_domain_name": main_name,
                    "feature": item.source_feature,
                    "score": item.score,
                    "confidence": item.confidence,
                    "source": item.source_detector,
                    "raw_value": item.raw_value,
                }
            )

        if unresolved_count:
            self._logger.warning(
                "[WARN] evidence builder still has %s unresolved subdomains after fallback",
                unresolved_count,
            )
        requested_total = len(activated_subdomains)
        diagnostics = {
            "requested_subdomains": requested_total,
            "id_matched": id_matched,
            "label_fallback_matched": fallback_matched,
            "unresolved_subdomains": unresolved_count,
            "unresolved_rate": round(
                float(unresolved_count / requested_total) if requested_total else 0.0,
                6,
            ),
            "fallback_triggered": fallback_triggered,
            "graph_query_degraded": False,
            "fallback_query_degraded": fallback_degraded,
        }
        self._last_diagnostics = diagnostics
        return evidence_list, diagnostics

    def build(self, activated_subdomains: List[ActivatedSubDomain]) -> List[Dict[str, Any]]:
        evidence_list, _ = self.build_with_diagnostics(activated_subdomains)
        return evidence_list


evidence_builder = EvidenceBuilder()
