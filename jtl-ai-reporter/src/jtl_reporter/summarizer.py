import statistics

def build_ai_summary(parsed):
    """
    Build an AI-like summary from parsed JTL metrics.
    No external API calls → deterministic, testable.
    """

    total_samples = parsed.get("sample_count", 0)
    avg = parsed.get("avg_elapsed", 0)
    max_ = parsed.get("max_elapsed", 0)
    min_ = parsed.get("min_elapsed", 0)
    errors = parsed.get("error_count", 0)
    throughput = parsed.get("throughput", 0)

    # Simple rule-based AI-style analysis
    summary = []

    summary.append(f"Total requests processed: {total_samples}.")
    summary.append(f"Average response time was {avg} ms, "
                   f"ranging from {min_} ms to {max_} ms.")

    if avg > 2000:
        summary.append("⚠️ Average latency is high — possible performance bottleneck.")
    elif avg > 1000:
        summary.append("⚠️ API performance is moderate and may need optimization.")
    else:
        summary.append("✅ API response time is healthy.")

    if errors > 0:
        summary.append(f"❗ There were {errors} failed requests.")
    else:
        summary.append("✅ No request failures detected.")

    if throughput < 5:
        summary.append("⚠️ Low throughput — load capacity may be limited.")
    else:
        summary.append("Throughput is acceptable.")

    return " ".join(summary)
