import os
from jtl_reporter.jtl_reporter import parse_jtl

BASE = os.path.dirname(__file__)

def test_parse_xml_success():
    file_path = os.path.join(BASE, "samples", "sample.xml")
    result = parse_jtl(file_path)

    assert result["count"] == 2
    assert result["avg_elapsed"] == 136.5
    assert result["max_elapsed"] == 150


def test_parse_xml_invalid():
    file_path = os.path.join(BASE, "samples", "invalid.xml")

    try:
        parse_jtl(file_path)
        assert False, "Should have failed for invalid XML"
    except Exception:
        assert True
