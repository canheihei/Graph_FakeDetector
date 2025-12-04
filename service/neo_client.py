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

def create_features_and_relations(tx, domain_name: str, features: list):
    """
    在事务中：
    1. 确保 SpecificDomain 节点存在（不存在则创建）
    2. 为每个 feature 创建 SubDomain 节点
    3. 建立 (:SubDomain)-[:SPECIFIC_OF]->(:SpecificDomain) 关系
    """
    # 1. 确保 Domain 节点存在（自动创建）
    tx.run(
        "MERGE (d:SpecificDomain {name: $domain_name})",
        domain_name=domain_name
    )

    # 2. 为每个 feature 创建节点并建立关系
    for feat in features:
        tx.run(
            """
            MATCH (d:SpecificDomain {name: $domain_name})
            MERGE (f:SubDomain {
                name: $name,
                describe: $describe,
                fake_score: $fake_score
            })
            MERGE (f)-[:SPECIFIC_OF]->(d)
            """,
            domain_name=domain_name,
            name=feat["name"],
            describe=feat["describe"],
            fake_score=feat["fake_score"]
        )

def apply_cyphers(cy_list):
    with driver.session() as session:
        for cy in cy_list:
            session.run(cy)


def process_result(result):
    with driver.session() as session:
        session.execute_write(
            create_features_and_relations,
            domain_name=result['domain'],
            features=result['features']
        )