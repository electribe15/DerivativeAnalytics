import pytest
from src.graph.neo4j_knowledge_graph import Neo4jKnowledgeGraph

@pytest.fixture
def setup_neo4j_graph():
    graph = Neo4jKnowledgeGraph()
    graph.connect()  # Assuming there's a connect method to establish a connection
    yield graph
    graph.close()  # Assuming there's a close method to clean up

def test_add_regulation(setup_neo4j_graph):
    graph = setup_neo4j_graph
    regulation_data = {
        "name": "EU Taxonomy Regulation",
        "description": "A regulation to guide sustainable investments.",
        "type": "regulation"
    }
    graph.add_regulation(regulation_data)  # Assuming there's an add_regulation method
    assert graph.get_regulation("EU Taxonomy Regulation") is not None

def test_link_regulation_to_policy(setup_neo4j_graph):
    graph = setup_neo4j_graph
    regulation_data = {
        "name": "EU Taxonomy Regulation",
        "description": "A regulation to guide sustainable investments.",
        "type": "regulation"
    }
    policy_data = {
        "name": "Company Carbon Emission Policy",
        "description": "Policy outlining carbon emission targets.",
        "type": "policy"
    }
    graph.add_regulation(regulation_data)
    graph.add_policy(policy_data)  # Assuming there's an add_policy method
    graph.link_regulation_to_policy("EU Taxonomy Regulation", "Company Carbon Emission Policy")  # Assuming there's a linking method
    assert graph.is_linked("EU Taxonomy Regulation", "Company Carbon Emission Policy")  # Assuming there's an is_linked method

def test_query_compliance_gaps(setup_neo4j_graph):
    graph = setup_neo4j_graph
    gaps = graph.query_compliance_gaps()  # Assuming there's a method to query compliance gaps
    assert isinstance(gaps, list)  # Expecting a list of gaps
    # Additional assertions can be added based on expected output

def test_remove_regulation(setup_neo4j_graph):
    graph = setup_neo4j_graph
    regulation_data = {
        "name": "EU Taxonomy Regulation",
        "description": "A regulation to guide sustainable investments.",
        "type": "regulation"
    }
    graph.add_regulation(regulation_data)
    graph.remove_regulation("EU Taxonomy Regulation")  # Assuming there's a remove_regulation method
    assert graph.get_regulation("EU Taxonomy Regulation") is None