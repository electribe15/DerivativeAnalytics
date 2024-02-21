import pytest
from fastapi.testclient import TestClient
from src.app import app

client = TestClient(app)

def test_get_compliance_status():
    response = client.get("/api/compliance/status")
    assert response.status_code == 200
    assert "status" in response.json()

def test_get_esg_report():
    response = client.get("/api/compliance/report")
    assert response.status_code == 200
    assert "report" in response.json()

def test_post_esg_data():
    data = {
        "company_name": "Test Company",
        "carbon_emissions": 1000,
        "supply_chain_ethics": "Compliant",
        "governance_score": 85
    }
    response = client.post("/api/compliance/data", json=data)
    assert response.status_code == 201
    assert response.json()["message"] == "Data submitted successfully"

def test_invalid_post_esg_data():
    data = {
        "company_name": "",
        "carbon_emissions": -100,
        "supply_chain_ethics": "Non-Compliant",
        "governance_score": 110
    }
    response = client.post("/api/compliance/data", json=data)
    assert response.status_code == 422  # Unprocessable Entity for invalid data