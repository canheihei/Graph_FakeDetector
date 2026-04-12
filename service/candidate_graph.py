from __future__ import annotations

import re
from typing import Any, Dict, Iterable


def _normalize_key(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(value or "").strip().lower()).strip("_")


class CandidateGraphStore:
    def __init__(self, neo4j_client) -> None:
        self._neo4j_client = neo4j_client

    def persist_candidates(self, items: Iterable[Dict[str, Any]]) -> None:
        serialized = list(items)
        if not serialized:
            return
        self._neo4j_client.execute_write(self._write_candidates, items=serialized)

    def update_status(self, candidate_ids: Iterable[str], status: str) -> None:
        serialized = [str(item) for item in candidate_ids if str(item).strip()]
        if not serialized:
            return
        self._neo4j_client.execute_write(
            self._update_candidate_status,
            candidate_ids=serialized,
            status=str(status or "").strip() or "pending",
        )

    @staticmethod
    def _write_candidates(tx, *, items):
        for item in items:
            graph = item.get("graph_candidate", {}) or {}
            source = item.get("source", {}) or {}
            main_name = str(graph.get("main_domain", "候选域泛化") or "候选域泛化")
            specific_name = str(graph.get("specific_domain", "未分类候选域") or "未分类候选域")
            subdomain_name = str(graph.get("subdomain_name", "未命名候选节点") or "未命名候选节点")
            main_key = _normalize_key(main_name)
            specific_key = f"{main_key}::{_normalize_key(specific_name)}"

            tx.run(
                """
                MERGE (m:CandidateMainDomain {candidate_main_key: $main_key})
                ON CREATE SET
                    m.name = $main_name,
                    m.status = $status
                ON MATCH SET
                    m.name = $main_name

                MERGE (s:CandidateSpecificDomain {candidate_specific_key: $specific_key})
                ON CREATE SET
                    s.name = $specific_name,
                    s.status = $status
                ON MATCH SET
                    s.name = $specific_name

                MERGE (s)-[:CANDIDATE_KINDS_OF]->(m)

                MERGE (sub:CandidateSubDomain {candidate_id: $candidate_id})
                SET sub.name = $subdomain_name,
                    sub.display_name = $subdomain_name,
                    sub.canonical_name = $canonical_name,
                    sub.describe = $describe,
                    sub.status = $status,
                    sub.source_type = $source_type,
                    sub.sample_name = $sample_name,
                    sub.detector = $detector,
                    sub.feature = $feature
                MERGE (sub)-[:CANDIDATE_SPECIFIC_OF]->(s)
                """,
                main_key=main_key,
                main_name=main_name,
                specific_key=specific_key,
                specific_name=specific_name,
                candidate_id=str(item.get("candidate_id")),
                subdomain_name=subdomain_name,
                canonical_name=str(graph.get("canonical_name", "") or ""),
                describe=str(graph.get("describe", "") or ""),
                status=str(item.get("status", "pending") or "pending"),
                source_type=str(source.get("source_type", "") or ""),
                sample_name=str(source.get("sample_name", "") or ""),
                detector=str((item.get("mapping_candidate", {}) or {}).get("detector", "") or ""),
                feature=str((item.get("mapping_candidate", {}) or {}).get("feature", "") or ""),
            )

    @staticmethod
    def _update_candidate_status(tx, *, candidate_ids, status):
        tx.run(
            """
            MATCH (sub:CandidateSubDomain)
            WHERE sub.candidate_id IN $candidate_ids
            SET sub.status = $status
            """,
            candidate_ids=candidate_ids,
            status=status,
        )
