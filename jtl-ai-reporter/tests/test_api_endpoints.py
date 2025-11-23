from fastapi.testclient import TestClient
from jtl_reporter.main import app

client = TestClient(app)


def test_upload_xml_success(client):
    file_data = open("tests/samples/sample.xml", "rb")

    response = client.post(
        "/api/v1/upload-jtl",
        files={"file": ("sample.xml", file_data, "application/xml")}
    )

    assert response.status_code == 200
    data = response.json()

    assert "parsed_report" in data
    assert "ai_summary" in data
    assert isinstance(data["ai_summary"], str)


def test_upload_invalid_xml(tmp_path):
    # Invalid XML content
    content = "<broken>"
    file_path = tmp_path / "invalid.xml"
    file_path.write_text(content)

    with open(file_path, "rb") as f:
        response = client.post("/api/v1/parse", files={"file": ("invalid.xml", f, "application/xml")})

    assert response.status_code == 400
    assert "Invalid JTL" in response.json()["detail"]
