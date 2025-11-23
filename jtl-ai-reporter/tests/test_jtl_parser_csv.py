import os
from jtl_reporter.jtl_reporter import parse_jtl

BASE = os.path.dirname(__file__)

def test_parse_csv_success():
    file_path = os.path.join(BASE, "samples", "sample.csv")
    result = parse_jtl(file_path)

    assert result["count"] == 2
    assert result["avg_elapsed"] == 136.5
    assert result["max_elapsed"] == 150
