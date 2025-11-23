import os
from fastapi.testclient import TestClient
from jtl_reporter.main import app

client = TestClient(app)


def test_upload_xml_success(tmp_path):
    # Create a temporary valid XML JTL file
    content = """<?xml version="1.0"?>
    <testResults>
        <httpSample t="120" />
        <httpSample t="150" />
    </testResults>
    """
    file_path = tmp_path / "sample.xml"
    file_path.write_text(content)

    with open(file_path, "rb") as f:
        response = client.post("/api/v1/parse", files={"file": ("sample.xml", f, "application/xml")})

    assert response.status_code == 200
    data = response.json()
    assert data["count"] == 2
    assert data["max_elapsed"] == 150
    assert data["avg_elapsed"] == 135


def test_upload_invalid_xml(tmp_path):
    # Invalid XML content
    content = "<broken>"
    file_path = tmp_path / "invalid.xml"
    file_path.write_text(content)

    with open(file_path, "rb") as f:
        response = client.post("/api/v1/parse", files={"file": ("invalid.xml", f, "application/xml")})

    assert response.status_code == 400
    assert "Invalid JTL" in response.json()["detail"]
