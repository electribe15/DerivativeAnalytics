from neo4j import GraphDatabase

class Neo4jKnowledgeGraph:
    def __init__(self, uri, user, password):
        self.driver = GraphDatabase.driver(uri, auth=(user, password))

    def close(self):
        self.driver.close()

    def create_relationship(self, regulation, policy, benchmark):
        with self.driver.session() as session:
            session.write_transaction(self._create_and_link, regulation, policy, benchmark)

    @staticmethod
    def _create_and_link(tx, regulation, policy, benchmark):
        tx.run("MERGE (r:Regulation {name: $regulation}) "
               "MERGE (p:Policy {name: $policy}) "
               "MERGE (b:Benchmark {name: $benchmark}) "
               "MERGE (r)-[:GOVERNS]->(p) "
               "MERGE (p)-[:ALIGNS_WITH]->(b)",
               regulation=regulation, policy=policy, benchmark=benchmark)

    def query_compliance(self, regulation):
        with self.driver.session() as session:
            result = session.read_transaction(self._find_compliance, regulation)
            return result

    @staticmethod
    def _find_compliance(tx, regulation):
        query = (
            "MATCH (r:Regulation {name: $regulation})-[:GOVERNS]->(p:Policy) "
            "RETURN p.name AS policy"
        )
        result = tx.run(query, regulation=regulation)
        return [record["policy"] for record in result]