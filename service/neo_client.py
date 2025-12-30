from neo4j import GraphDatabase
from config import NEO4J_URI, NEO4J_USER, NEO4J_PASS

driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASS))

def get_existing_schema(prompt):
    with driver.session() as session:
        query = """
        MATCH (x)-[r]->(d:Domain {name:$name})
        RETURN d, collect(x) AS related

        """
        result = session.run(query, name=prompt).data()
        return result or {}


def get_specificdomain():
    """
    查询连接到MainDomain节点的所有节点
    """
    try:
        with driver.session() as session:
            query = """
            MATCH (x)-[r]->(:MainDomain)
            RETURN x
            """
            result = session.run(query)

            # 将结果转换为列表
            records = list(result)
            if records:
                # 提取节点数据
                nodes = [dict(record["x"]) for record in records]
                return {"data": nodes, "count": len(nodes)}
            else:
                return {"data": [], "count": 0}

    except Exception as e:
        return {"error": str(e), "data": []}


def get_subdomain():
    """
        查询连接到SUbDomain节点的所有节点
    """
    try:
        with driver.session() as session:
            query = """
            MATCH (x)-[r]->(:SpecificDomain)
            RETURN x
            """
            result = session.run(query)
            # 将结果转换为列表
            records = list(result)
            if records:
                # 提取节点数据
                nodes = [dict(record["x"]) for record in records]
                return {"data": nodes, "count": len(nodes)}
            else:
                return {"data": [], "count": 0}

    except Exception as e:
        return {"error": str(e), "data": []}

def create_features_and_relations(tx, specific_domain: str, describe: str, specific_id: str, subdomain: list):
    """
    在事务中：
    1. 确保 SpecificDomain 节点存在（不存在则创建）
    2. 为每个子特征创建 SubDomain 节点
    3. 建立 (:SubDomain)-[:SPECIFIC_OF]->(:SpecificDomain) 关系
    """
    # 1. 确保 SpecificDomain 节点存在（基于 name 唯一标识）
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
        specific_domain=specific_domain,
        describe=describe,
        specific_id=specific_id
    )

    # 2. 为每个子特征创建 SubDomain 节点，并关联到 SpecificDomain
    for sub in subdomain:
        tx.run(
            """
            MATCH (d:SpecificDomain {name: $specific_domain})
            MERGE (f:SubDomain {
                name: $name,
                describe: $describe,
                sub_id: $sub_id
            })
            MERGE (f)-[:SPECIFIC_OF]->(d)
            """,
            specific_domain=specific_domain,
            name=sub["name"],
            describe=sub["describe"],
            sub_id=sub["sub_id"]
        )

def apply_cyphers(cy_list):
    with driver.session() as session:
        for cy in cy_list:
            session.run(cy)


def process_result(result):
    with driver.session() as session:
        session.execute_write(
            create_features_and_relations,
            specific_domain=result['specific_domain'],
            describe=result['describe'],
            specific_id=result['specific_id'],
            subdomain=result['subdomain']
        )


from neo4j import GraphDatabase
from config import NEO4J_URI, NEO4J_USER, NEO4J_PASS

class Neo4jClient:

    def __init__(self):
        self.driver = GraphDatabase.driver(
            NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASS)
        )

    def query(self, cypher, params=None):
        with self.driver.session() as session:
            result = session.run(cypher, params or {})
            return [record.data() for record in result]

neo4j_client = Neo4jClient()
