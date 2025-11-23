from jtl_reporter.summarizer import build_ai_summary

def test_ai_summary_basic():
    parsed = {
        "sample_count": 100,
        "avg_elapsed": 120,
        "max_elapsed": 500,
        "min_elapsed": 30,
        "error_count": 0,
        "throughput": 10
    }

    summary = build_ai_summary(parsed)

    assert "Total requests processed: 100" in summary
    assert "Average response time was 120 ms" in summary
    assert "healthy" in summary.lower()
    assert "no request failures" in summary.lower()
