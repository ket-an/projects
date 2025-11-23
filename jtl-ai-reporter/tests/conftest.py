import pytest
from fastapi.testclient import TestClient
from jtl_reporter.main import app

@pytest.fixture
def client():
    return TestClient(app)