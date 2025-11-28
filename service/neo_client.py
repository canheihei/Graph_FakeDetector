from neo4j import GraphDatabase
from config import NEO4J_URI, NEO4J_USER, NEO4J_PASS

driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASS))

def get_existing_schema(prompt):
    with driver.session() as session:
        query = """
        MATCH (d:Domain {name:$name})-[*]-(x)
        RETURN d, collect(x) as related
        """
        result = session.run(query, name=prompt).data()
        return result or {}

def apply_cyphers(cy_list):
    with driver.session() as session:
        for cy in cy_list:
            session.run(cy)
