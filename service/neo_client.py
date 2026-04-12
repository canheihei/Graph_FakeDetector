from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from neo4j import GraphDatabase

from config import NEO4J_PASS, NEO4J_URI, NEO4J_USER
from service.graph_semantics import match_existing_subdomain


driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASS))


@dataclass(frozen=True)
class GraphWritePayload:
    main_domain: Optional[str]
    main_describe: str
    specific_domain: str
    describe: str
    specific_id: str
    subdomain: List[Dict[str, str]]
    semantic_source: str = "unknown"
    semantic_prompt: str = ""
    semantic_version: str = "graph_semantics_v1"

    @classmethod
    def from_dict(cls, payload: Dict) -> "GraphWritePayload":
        return cls(
            main_domain=payload.get("main_domain"),
            main_describe=payload.get("main_describe", ""),
            specific_domain=payload["specific_domain"],
            describe=payload["describe"],
            specific_id=payload["specific_id"],
            subdomain=list(payload["subdomain"]),
            semantic_source=payload.get("semantic_source", "unknown"),
            semantic_prompt=payload.get("semantic_prompt", ""),
            semantic_version=payload.get("semantic_version", "graph_semantics_v1"),
        )


@dataclass(frozen=True)
class SemanticDedupPlan:
    similar_specific_domain: Optional[str] = None
    similar_subdomains: Dict[str, Dict[str, str]] = field(default_factory=dict)
    matched_subdomains: int = 0
    created_subdomains: int = 0
    semantic_threshold: float = 0.80

    @property
    def target_specific_domain(self) -> Optional[str]:
        return self.similar_specific_domain


class GraphWriteStrategy(ABC):
    @abstractmethod
    def write(
        self,
        tx,
        payload: GraphWritePayload,
        plan: Optional[SemanticDedupPlan] = None,
    ) -> None:
        raise NotImplementedError


class PlainGraphWriteStrategy(GraphWriteStrategy):
    def write(
        self,
        tx,
        payload: GraphWritePayload,
        plan: Optional[SemanticDedupPlan] = None,
    ) -> None:
        self._merge_specific_domain(
            tx=tx,
            main_domain=payload.main_domain,
            main_describe=payload.main_describe,
            name=payload.specific_domain,
            describe=payload.describe,
            specific_id=payload.specific_id,
        )
        self._merge_subdomains(
            tx=tx,
            specific_domain=payload.specific_domain,
            subdomains=payload.subdomain,
            semantic_source=payload.semantic_source,
            semantic_version=payload.semantic_version,
        )

    @staticmethod
    def _merge_specific_domain(
        tx,
        main_domain: Optional[str],
        main_describe: str,
        name: str,
        describe: str,
        specific_id: str,
    ) -> None:
        if main_domain:
            tx.run(
                """
                MERGE (d:SpecificDomain {name: $specific_domain})
                ON CREATE SET
                    d.specific_id = $specific_id,
                    d.describe = $describe
                ON MATCH SET
                    d.specific_id = $specific_id,
                    d.describe = CASE
                        WHEN coalesce(d.describe, '') = '' THEN $describe
                        ELSE d.describe
                    END
                WITH d
                OPTIONAL MATCH (d)-[:KINDS_OF]->(existing_main:MainDomain)
                WITH d, existing_main
                FOREACH (_ IN CASE WHEN existing_main IS NULL AND $main_domain <> '' THEN [1] ELSE [] END |
                    MERGE (m:MainDomain {name: $main_domain})
                    ON CREATE SET m.describe = $main_describe
                    ON MATCH SET m.describe = CASE
                        WHEN coalesce(m.describe, '') = '' THEN $main_describe
                        ELSE m.describe
                    END
                    MERGE (d)-[:KINDS_OF]->(m)
                )
                """,
                main_domain=main_domain,
                main_describe=main_describe,
                specific_domain=name,
                describe=describe,
                specific_id=specific_id,
            )
            return

        tx.run(
            """
            MERGE (d:SpecificDomain {name: $specific_domain})
            ON CREATE SET
                d.specific_id = $specific_id,
                d.describe = $describe
            ON MATCH SET
                d.specific_id = $specific_id,
                d.describe = $describe
            """,
            specific_domain=name,
            describe=describe,
            specific_id=specific_id,
        )

    @staticmethod
    def _merge_subdomains(
        tx,
        specific_domain: str,
        subdomains: List[Dict[str, str]],
        semantic_source: str,
        semantic_version: str,
    ) -> None:
        for sub in subdomains:
            tx.run(
                """
                MATCH (d:SpecificDomain {name: $specific_domain})
                MERGE (f:SubDomain {sub_id: $sub_id})
                ON CREATE SET
                    f.name = $name,
                    f.display_name = $display_name,
                    f.canonical_name = $canonical_name,
                    f.describe = $describe,
                    f.semantic_source = $semantic_source,
                    f.semantic_version = $semantic_version
                ON MATCH SET
                    f.name = $name,
                    f.display_name = coalesce($display_name, f.display_name, $name),
                    f.canonical_name = coalesce($canonical_name, f.canonical_name),
                    f.describe = CASE
                        WHEN coalesce(f.describe, '') = '' THEN $describe
                        ELSE f.describe
                    END,
                    f.semantic_source = coalesce(f.semantic_source, $semantic_source),
                    f.semantic_version = $semantic_version
                MERGE (f)-[:SPECIFIC_OF]->(d)
                """,
                specific_domain=specific_domain,
                name=sub["name"],
                display_name=sub.get("display_name", sub["name"]),
                canonical_name=sub.get("canonical_name"),
                describe=sub["describe"],
                sub_id=sub["sub_id"],
                semantic_source=semantic_source,
                semantic_version=semantic_version,
            )


class SemanticDedupGraphWriteStrategy(PlainGraphWriteStrategy):
    def write(
        self,
        tx,
        payload: GraphWritePayload,
        plan: Optional[SemanticDedupPlan] = None,
    ) -> None:
        if plan is None:
            raise ValueError("SemanticDedupGraphWriteStrategy requires a dedup plan")

        target_specific_domain = plan.target_specific_domain or payload.specific_domain

        if plan.target_specific_domain:
            print(
                f"[REUSE] SpecificDomain: '{payload.specific_domain}' -> "
                f"'{plan.target_specific_domain}'"
            )
            tx.run(
                """
                MATCH (d:SpecificDomain {name: $name})
                OPTIONAL MATCH (d)-[:KINDS_OF]->(m:MainDomain)
                WITH d, m
                FOREACH (_ IN CASE WHEN m IS NULL AND $main_domain <> '' THEN [1] ELSE [] END |
                    MERGE (target_main:MainDomain {name: $main_domain})
                    ON CREATE SET target_main.describe = $main_describe
                    MERGE (d)-[:KINDS_OF]->(target_main)
                )
                SET d.describe = CASE
                    WHEN coalesce(d.describe, '') = '' THEN $describe
                    ELSE d.describe
                END
                """,
                name=plan.target_specific_domain,
                describe=payload.describe,
                main_domain=payload.main_domain or "",
                main_describe=payload.main_describe,
            )
        else:
            print(f"[CREATE] SpecificDomain: '{payload.specific_domain}'")
            self._merge_specific_domain(
                tx=tx,
                main_domain=payload.main_domain,
                main_describe=payload.main_describe,
                name=payload.specific_domain,
                describe=payload.describe,
                specific_id=payload.specific_id,
            )

        for sub in payload.subdomain:
            similar_sub = plan.similar_subdomains.get(sub["name"])
            if similar_sub:
                similar_name = similar_sub.get("name", sub["name"])
                similar_sub_id = similar_sub.get("sub_id", "")
                print(f"[REUSE] SubDomain: '{sub['name']}' -> '{similar_name}'")
                tx.run(
                    """
                    MATCH (sub:SubDomain)-[:SPECIFIC_OF]->(s:SpecificDomain {name: $specific_domain})
                    WHERE ($similar_sub_id <> '' AND sub.sub_id = $similar_sub_id)
                       OR ($similar_sub_id = '' AND sub.name = $similar_name)
                    SET sub.describe = CASE
                            WHEN coalesce(sub.describe, '') = '' THEN $describe
                            ELSE sub.describe
                        END,
                        sub.display_name = coalesce(sub.display_name, $display_name, sub.name),
                        sub.canonical_name = coalesce(sub.canonical_name, $canonical_name),
                        sub.semantic_source = coalesce(sub.semantic_source, $semantic_source),
                        sub.semantic_version = $semantic_version
                    """,
                    similar_name=similar_name,
                    similar_sub_id=similar_sub_id,
                    specific_domain=target_specific_domain,
                    describe=sub["describe"],
                    display_name=sub.get("display_name", sub["name"]),
                    canonical_name=sub.get("canonical_name"),
                    semantic_source=payload.semantic_source,
                    semantic_version=payload.semantic_version,
                )
                continue

            print(f"[CREATE] SubDomain: '{sub['name']}' under '{target_specific_domain}'")
            self._merge_subdomains(
                tx=tx,
                specific_domain=target_specific_domain,
                subdomains=[sub],
                semantic_source=payload.semantic_source,
                semantic_version=payload.semantic_version,
            )


class Neo4jClient:
    _SPECIFIC_DOMAIN_QUERY = """
        MATCH (s:SpecificDomain)
        RETURN s.specific_id AS id, s.name AS name, s.describe AS describe
    """

    _SPECIFIC_DOMAIN_WITH_MAIN_QUERY = """
        MATCH (s:SpecificDomain)
        OPTIONAL MATCH (s)-[:KINDS_OF]->(m:MainDomain)
        RETURN s.specific_id AS id, s.name AS name, s.describe AS describe,
               m.name AS main_domain, m.describe AS main_describe
    """

    _SUBDOMAIN_QUERY = """
        MATCH (s:SubDomain)
        RETURN s.sub_id AS id, s.name AS name
    """

    def __init__(self):
        self.driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASS))

    def query(self, cypher: str, params: Optional[Dict] = None) -> List[Dict]:
        with self.driver.session() as session:
            result = session.run(cypher, params or {})
            return [record.data() for record in result]

    @staticmethod
    def _node_to_dict(value):
        if isinstance(value, dict):
            return value
        try:
            return dict(value)
        except TypeError:
            return value

    def execute_write(self, fn, **kwargs):
        with self.driver.session() as session:
            return session.execute_write(fn, **kwargs)

    def get_existing_schema(self, prompt: str):
        return self.query(
            """
            MATCH (x)-[r]->(d:Domain {name:$name})
            RETURN d, collect(x) AS related
            """,
            {"name": prompt},
        ) or {}

    def get_specificdomain_nodes(self) -> Dict:
        try:
            records = self.query(
                """
                MATCH (x:SpecificDomain)
                RETURN x
                """
            )
            nodes = [self._node_to_dict(record["x"]) for record in records]
            return {"data": nodes, "count": len(nodes)}
        except Exception as exc:
            return {"error": str(exc), "data": []}

    def get_subdomain_nodes(self) -> Dict:
        try:
            records = self.query(
                """
                MATCH (x:SubDomain)-[r]->(:SpecificDomain)
                RETURN x
                """
            )
            nodes = [self._node_to_dict(record["x"]) for record in records]
            return {"data": nodes, "count": len(nodes)}
        except Exception as exc:
            return {"error": str(exc), "data": []}

    def list_specific_domains(self, include_main_domain: bool = False) -> List[Dict]:
        query = (
            self._SPECIFIC_DOMAIN_WITH_MAIN_QUERY
            if include_main_domain
            else self._SPECIFIC_DOMAIN_QUERY
        )
        return self.query(query)

    def list_subdomains(self) -> List[Dict]:
        return self.query(self._SUBDOMAIN_QUERY)

    def list_main_domains(self) -> List[Dict]:
        return self.query(
            """
            MATCH (m:MainDomain)
            RETURN m.name AS name, m.describe AS describe
            """
        )

    def list_specific_domain_names(self) -> List[str]:
        records = self.query("MATCH (s:SpecificDomain) RETURN s.name AS name")
        return [item["name"] for item in records]

    def list_subdomain_names(self, specific_domain_name: str) -> List[str]:
        records = self.query(
            """
            MATCH (sub:SubDomain)-[:SPECIFIC_OF]->(s:SpecificDomain {name: $name})
            RETURN sub.name AS name
            """,
            {"name": specific_domain_name},
        )
        return [item["name"] for item in records]

    def list_subdomain_records(self, specific_domain_name: str) -> List[Dict]:
        return self.query(
            """
            MATCH (sub:SubDomain)-[:SPECIFIC_OF]->(s:SpecificDomain {name: $name})
            OPTIONAL MATCH (s)-[:KINDS_OF]->(m:MainDomain)
            RETURN sub.sub_id AS sub_id,
                   sub.name AS name,
                   coalesce(sub.display_name, sub.name) AS display_name,
                   sub.canonical_name AS canonical_name,
                   sub.describe AS describe,
                   s.name AS specific_domain,
                   m.name AS main_domain
            """,
            {"name": specific_domain_name},
        )

    def find_subdomain_record(
        self,
        *,
        specific_domain_name: str,
        sub_id: str = "",
        canonical_name: str = "",
        sub_name: str = "",
    ) -> Optional[Dict]:
        records = self.list_subdomain_records(specific_domain_name)
        if sub_id:
            matched = next(
                (item for item in records if str(item.get("sub_id", "")) == str(sub_id)),
                None,
            )
            if matched is not None:
                return matched

        normalized_canonical = str(canonical_name or "").strip().lower()
        if normalized_canonical:
            matched = next(
                (
                    item for item in records
                    if str(item.get("canonical_name", "") or "").strip().lower() == normalized_canonical
                ),
                None,
            )
            if matched is not None:
                return matched

        normalized_name = str(sub_name or "").strip()
        if normalized_name:
            matched = next(
                (
                    item for item in records
                    if normalized_name in {
                        str(item.get("name", "") or "").strip(),
                        str(item.get("display_name", "") or "").strip(),
                    }
                ),
                None,
            )
            if matched is not None:
                return matched

        return None

    def get_specific_domain_by_subdomain_name(self, sub_name: str) -> Optional[Dict]:
        result = self.query(
            """
            MATCH (sub:SubDomain {name: $sub_name})-[:SPECIFIC_OF]->(s:SpecificDomain)
            OPTIONAL MATCH (s)-[:KINDS_OF]->(m:MainDomain)
            RETURN s.specific_id AS id, s.name AS name, s.describe AS describe,
                   m.name AS main_domain, m.describe AS main_describe
            """,
            {"sub_name": sub_name},
        )
        return result[0] if result else None

    def get_main_domain_name_by_specific_domain(self, specific_domain_name: str) -> Optional[str]:
        result = self.query(
            """
            MATCH (s:SpecificDomain {name: $name})-[:KINDS_OF]->(m:MainDomain)
            RETURN m.name AS name
            """,
            {"name": specific_domain_name},
        )
        return result[0]["name"] if result else None

    def get_main_domain_describe(self, main_domain_name: str) -> Optional[str]:
        result = self.query(
            """
            MATCH (m:MainDomain {name: $name})
            RETURN m.describe AS describe
            """,
            {"name": main_domain_name},
        )
        return result[0]["describe"] if result else None

    def get_graph_stats(self) -> Dict:
        stats = {
            "node_counts": self.query(
                """
                MATCH (n)
                RETURN labels(n)[0] AS label, count(n) AS count
                """
            ),
            "relation_counts": self.query(
                """
                MATCH ()-[r]->()
                RETURN type(r) AS type, count(r) AS count
                """
            ),
            "domain_structure": self.query(
                """
                MATCH (s:SubDomain)-[:SPECIFIC_OF]->(d:SpecificDomain)
                RETURN d.name AS domain, count(s) AS sub_count
                """
            ),
            "subdomain_list": self.query(
                """
                MATCH (s:SubDomain)
                RETURN coalesce(s.display_name, s.name) AS name, s.sub_id AS id
                ORDER BY s.sub_id
                """
            ),
        }
        graph_overview = self.query(
            """
            MATCH (n)
            WITH count(n) AS nodes
            MATCH ()-[r]->()
            RETURN nodes, count(r) AS relations
            """
        )
        stats["graph_overview"] = graph_overview[0] if graph_overview else {}
        return stats

    def get_graph_overview(self) -> Dict:
        main_domains = self.query(
            """
            MATCH (m:MainDomain)
            RETURN m.name AS name, m.describe AS describe
            """
        )
        specific_domains = self.query(self._SPECIFIC_DOMAIN_WITH_MAIN_QUERY)
        sub_domains = self.query(
            """
            MATCH (sub:SubDomain)-[:SPECIFIC_OF]->(s:SpecificDomain)
            OPTIONAL MATCH (s)-[:KINDS_OF]->(m:MainDomain)
            RETURN sub.sub_id AS id,
                   coalesce(sub.display_name, sub.name) AS name,
                   sub.name AS raw_name,
                   sub.canonical_name AS canonical_name,
                   sub.describe AS describe,
                   s.name AS specific_domain,
                   coalesce(m.name, '未连接主域') AS main_domain
            """
        )
        return {
            "main_domains": main_domains,
            "specific_domains": specific_domains,
            "sub_domains": sub_domains,
            "summary": {
                "main_domain_count": len(main_domains),
                "specific_domain_count": len(specific_domains),
                "sub_domain_count": len(sub_domains),
            },
        }


class GraphResultWriter:
    def __init__(self, client: Neo4jClient):
        self.client = client
        self._plain_strategy = PlainGraphWriteStrategy()
        self._semantic_strategy = SemanticDedupGraphWriteStrategy()

    def write(self, result: Dict, semantic_threshold: Optional[float] = None):
        payload = GraphWritePayload.from_dict(result)
        if semantic_threshold is None:
            self.client.execute_write(self._plain_strategy.write, payload=payload)
            return None

        plan = self._build_dedup_plan(payload, semantic_threshold)
        self.client.execute_write(
            self._semantic_strategy.write,
            payload=payload,
            plan=plan,
        )
        stats = {
            "specific_domain_reused": plan.similar_specific_domain is not None,
            "specific_domain_name": plan.target_specific_domain or payload.specific_domain,
            "subdomain_matched": plan.matched_subdomains,
            "subdomain_created": plan.created_subdomains,
            "total_subdomain": len(payload.subdomain),
            "semantic_threshold": semantic_threshold,
        }
        print(f"[REUSE] Semantic dedup summary: {stats}")
        return stats

    def _build_dedup_plan(
        self,
        payload: GraphWritePayload,
        semantic_threshold: float,
    ) -> SemanticDedupPlan:
        from service.llm_chain import semantic_match

        existing_specific_names = self.client.list_specific_domain_names()
        similar_specific = None

        if existing_specific_names:
            matched = semantic_match(
                payload.specific_domain,
                existing_specific_names,
                semantic_threshold,
            )
            if matched != payload.specific_domain:
                similar_specific = matched
                print(
                    f"[REUSE] Matched SpecificDomain "
                    f"'{payload.specific_domain}' -> '{matched}'"
                )
            else:
                print(f"[CREATE] New SpecificDomain candidate: '{payload.specific_domain}'")
        else:
            print("[CREATE] No existing SpecificDomain nodes found")

        target_specific = similar_specific or payload.specific_domain
        existing_subdomains = self.client.list_subdomain_records(target_specific)

        similar_subdomains: Dict[str, Dict[str, str]] = {}
        matched_count = 0
        created_count = 0
        for sub in payload.subdomain:
            matched_record = match_existing_subdomain(
                sub,
                existing_subdomains,
                semantic_threshold=max(semantic_threshold, 0.88),
            )
            if matched_record is not None:
                similar_subdomains[sub["name"]] = {
                    "name": matched_record.get("name") or sub["name"],
                    "sub_id": matched_record.get("sub_id") or "",
                }
                matched_count += 1
                print(
                    f"[REUSE] Matched SubDomain '{sub['name']}' -> "
                    f"'{matched_record.get('display_name') or matched_record.get('name')}'"
                )
                continue

            created_count += 1
            print(f"[CREATE] New SubDomain candidate: '{sub['name']}'")

        return SemanticDedupPlan(
            similar_specific_domain=similar_specific,
            similar_subdomains=similar_subdomains,
            matched_subdomains=matched_count,
            created_subdomains=created_count,
            semantic_threshold=semantic_threshold,
        )


neo4j_client = Neo4jClient()
graph_writer = GraphResultWriter(neo4j_client)


def get_existing_schema(prompt):
    return neo4j_client.get_existing_schema(prompt)


def get_specificdomain():
    return neo4j_client.get_specificdomain_nodes()


def get_subdomain():
    return neo4j_client.get_subdomain_nodes()


def apply_cyphers(cy_list):
    with driver.session() as session:
        for cy in cy_list:
            session.run(cy)


def process_result(result):
    return graph_writer.write(result)


def process_result_with_semantic_dedup(result, semantic_threshold: float = 0.80):
    return graph_writer.write(result, semantic_threshold=semantic_threshold)
