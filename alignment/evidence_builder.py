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
                if alias and alias not in label_lookup:
                    label_lookup[alias] = graph_info
        return label_lookup

    def build(self, activated_subdomains: List[ActivatedSubDomain]) -> List[Dict[str, Any]]:
        if not activated_subdomains:
            return []

        sub_ids = [item.subdomain_id for item in activated_subdomains]
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
            return []
        unresolved = [item for item in activated_subdomains if item.subdomain_id not in graph_lookup]

        fallback_lookup: Dict[str, Dict[str, Any]] = {}
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
                self._logger.info(
                    "[WARN] evidence lookup fallback triggered for %s subdomains",
                    len(unresolved),
                )
            except Exception as exc:
                if not self._graph_fallback_failed:
                    self._logger.warning(
                        "[WARN] evidence builder degraded: failed label fallback lookup: %s",
                        exc,
                    )
                    self._graph_fallback_failed = True

        unresolved_count = 0

        evidence_list: List[Dict[str, Any]] = []
        for item in activated_subdomains:
            graph_info = graph_lookup.get(item.subdomain_id) or fallback_lookup.get(item.subdomain_label, {})
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
        return evidence_list


evidence_builder = EvidenceBuilder()
