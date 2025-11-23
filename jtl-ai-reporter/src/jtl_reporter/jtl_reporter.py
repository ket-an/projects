"""
jtl_reporter.py

Standalone Python module to parse JMeter JTL (XML or CSV), clean columns,
compute metrics, detect simple anomalies, and emit a JSON report and
human-readable summary. Includes a CLI for quick usage.

Usage examples:
  python jtl_reporter.py --input example.jtl --out report.json --keep elapsed,label,success --format json
  python jtl_reporter.py --input example.csv --drop threadName,bytes --out report.csv --format csv

Replace generate_llm_summary(...) with your LLM call wrapper when ready.
"""

from __future__ import annotations
import argparse
import csv
import json
import os
from dataclasses import dataclass, asdict
from datetime import datetime
from typing import List, Dict, Any, Optional

import numpy as np
import pandas as pd
from scipy import stats
import xml.etree.ElementTree as ET


DEFAULT_KEEP_COLUMNS = ["timeStamp", "elapsed", "label", "success", "bytes", "threadName", "latency", "responseCode"]


@dataclass
class Report:
    generated_at: str
    metrics: Dict[str, Any]
    anomalies: Dict[str, Any]
    summary: str
    columns: List[str]

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2, default=str)


# -------------------- Parsing --------------------

def parse_jtl_xml(path: str) -> pd.DataFrame:
    """Parse an XML JTL file (JMeter standard) into a DataFrame.

    The code attempts to handle both <sample> and <httpSample> tag styles.
    """
    tree = ET.parse(path)
    root = tree.getroot()
    rows = []

    # Try common sample tags
    for tag in ("httpSample", "sample", "sample_result"):
        for elem in root.findall(f".//{tag}"):
            if elem is None:
                continue
            rows.append({k: elem.get(k) for k in elem.keys()})

    # If none found, try top level children
    if not rows:
        for child in root:
            rows.append({k: child.get(k) for k in child.keys()})

    df = pd.DataFrame(rows)
    return _normalize_types(df)


def parse_jtl_csv(path: str) -> pd.DataFrame:
    """Parse CSV JTL file. Handles header or headerless CSV produced by JMeter."""
    # Try reading with pandas; JMeter CSV usually has header but not guaranteed
    try:
        df = pd.read_csv(path)
    except Exception:
        # Fallback: try to infer columns by reading first row count
        with open(path, "r", newline="") as f:
            reader = csv.reader(f)
            first = next(reader)
            ncols = len(first)
        df = pd.read_csv(path, header=None)
        df.columns = [f"col{i}" for i in range(len(df.columns))]
    return _normalize_types(df)


def parse_jtl(path: str) -> pd.DataFrame:
    """Dispatch parser based on file extension."""
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    ext = os.path.splitext(path)[1].lower()
    if ext in (".xml", ".jtl"):
        return parse_jtl_xml(path)
    elif ext in (".csv", ".txt"):
        return parse_jtl_csv(path)
    else:
        # Try both
        try:
            return parse_jtl_xml(path)
        except Exception:
            return parse_jtl_csv(path)


def _normalize_types(df: pd.DataFrame) -> pd.DataFrame:
    """Convert common JTL columns to appropriate dtypes and compute datetime.

    Columns handled: timeStamp -> datetime, elapsed -> numeric, success -> bool
    """
    if df.empty:
        return df

    df = df.copy()
    # strip whitespace from column names
    df.columns = [c.strip() if isinstance(c, str) else c for c in df.columns]

    if "timeStamp" in df.columns:
        df["timeStamp"] = pd.to_numeric(df["timeStamp"], errors="coerce")
        df["datetime"] = pd.to_datetime(df["timeStamp"], unit="ms", errors="coerce")
    elif "time" in df.columns and df["time"].dtype == object:
        # sometimes column name is 'time' and already in ms
        df["timeStamp"] = pd.to_numeric(df["time"], errors="coerce")
        df["datetime"] = pd.to_datetime(df["timeStamp"], unit="ms", errors="coerce")

    if "elapsed" in df.columns:
        df["elapsed"] = pd.to_numeric(df["elapsed"], errors="coerce")
    if "latency" in df.columns:
        df["latency"] = pd.to_numeric(df["latency"], errors="coerce")
    if "bytes" in df.columns:
        df["bytes"] = pd.to_numeric(df["bytes"], errors="coerce")

    if "success" in df.columns:
        df["success"] = df["success"].astype(str).str.lower().map({"true": True, "false": False, "t": True, "f": False}).fillna(False)

    return df


# -------------------- Cleaning / Selection --------------------

def clean_dataframe(df: pd.DataFrame, keep: Optional[List[str]] = None, drop: Optional[List[str]] = None) -> pd.DataFrame:
    """Return DataFrame containing only keep columns (if provided) and dropping any drop columns.

    Keeps columns but preserves their existing order.
    """
    if df.empty:
        return df

    df = df.copy()
    if keep:
        keep_existing = [c for c in keep if c in df.columns]
        if not keep_existing:
            # if none of the keep columns present, fall back to original df
            pass
        else:
            df = df[keep_existing].copy()
    if drop:
        drop_existing = [c for c in drop if c in df.columns]
        if drop_existing:
            df = df.drop(columns=drop_existing, errors="ignore")
    return df


# -------------------- Metrics --------------------

def compute_metrics(df: pd.DataFrame) -> Dict[str, Any]:
    metrics: Dict[str, Any] = {}
    if df.empty:
        metrics.update({"count": 0})
        return metrics

    elapsed = df["elapsed"].dropna().astype(float) if "elapsed" in df.columns else pd.Series(dtype=float)
    metrics["count"] = int(len(df))
    if not elapsed.empty:
        metrics["avg_ms"] = float(elapsed.mean())
        metrics["min_ms"] = float(elapsed.min())
        metrics["max_ms"] = float(elapsed.max())
        metrics["p50_ms"] = float(elapsed.quantile(0.5))
        metrics["p90_ms"] = float(elapsed.quantile(0.9))
        metrics["p95_ms"] = float(elapsed.quantile(0.95))
        metrics["p99_ms"] = float(elapsed.quantile(0.99))
        metrics["stddev_ms"] = float(elapsed.std())
    else:
        for k in ["avg_ms","min_ms","max_ms","p50_ms","p90_ms","p95_ms","p99_ms","stddev_ms"]:
            metrics[k] = None

    # throughput
    if "datetime" in df.columns and not df["datetime"].dropna().empty:
        times = df["datetime"].sort_values()
        total_seconds = (times.iloc[-1] - times.iloc[0]).total_seconds() or 1
        metrics["throughput_rps"] = float(metrics["count"]) / float(total_seconds)
    else:
        metrics["throughput_rps"] = None

    # errors
    if "success" in df.columns:
        err_count = int((df["success"] == False).sum())
        metrics["error_count"] = err_count
        metrics["error_rate"] = float(err_count / metrics["count"]) if metrics["count"] > 0 else None
    else:
        metrics["error_count"] = None
        metrics["error_rate"] = None

    # top slow endpoints by p95 if label present
    if "label" in df.columns and "elapsed" in df.columns:
        grouped = df.groupby("label")["elapsed"].agg(["count", lambda s: float(np.percentile(s.dropna(), 95))])
        grouped.columns = ["count", "p95_ms"]
        top = grouped.sort_values("p95_ms", ascending=False).head(10)
        metrics["top_slowest_labels"] = top.reset_index().to_dict(orient="records")

    return metrics


# -------------------- Anomaly detection / highlights --------------------

def detect_anomalies(df: pd.DataFrame, z_thresh: float = 3.0) -> Dict[str, Any]:
    result: Dict[str, Any] = {"anomalies": [], "high_error_windows": []}
    if df.empty:
        return result

    if "elapsed" in df.columns and not df["elapsed"].dropna().empty:
        elapsed = df["elapsed"].astype(float).dropna()
        if len(elapsed) > 1:
            zscores = np.abs(stats.zscore(elapsed, nan_policy='omit'))
            anomalies_idx = np.where(zscores > z_thresh)[0]
            for idx in anomalies_idx:
                result["anomalies"].append({
                    "index": int(idx),
                    "value": float(elapsed.iloc[idx]),
                    "zscore": float(zscores[idx])
                })

    # error windows: resample per minute and find windows where error rate crosses a threshold
    if "datetime" in df.columns and "success" in df.columns:
        tmp = df.set_index("datetime").resample("1Min").agg({"success": lambda s: (s==False).sum(), "elapsed": "count"})
        tmp = tmp[tmp["elapsed"] > 0]
        if not tmp.empty:
            tmp["error_rate"] = tmp["success"] / tmp["elapsed"]
            high = tmp[tmp["error_rate"] > 0.2]  # configurable threshold
            if not high.empty:
                result["high_error_windows"] = high.reset_index().to_dict(orient="records")

    return result


# -------------------- Summary generation (template + LLM placeholder) --------------------

def generate_text_summary(metrics: Dict[str,Any], anomalies: Dict[str,Any]) -> str:
    lines: List[str] = []
    lines.append(f"Report generated at {datetime.utcnow().isoformat()}Z")
    lines.append(f"Total samples: {metrics.get('count', 0)}")
    if metrics.get("count", 0) > 0:
        avg = metrics.get("avg_ms")
        p95 = metrics.get("p95_ms")
        p99 = metrics.get("p99_ms")
        lines.append(f"Average latency: {avg:.2f} ms (p95: {p95:.2f} ms, p99: {p99:.2f} ms)" if avg and p95 and p99 else "Latency summary not available")
        tp = metrics.get("throughput_rps")
        lines.append(f"Throughput: {tp:.2f} req/s" if tp else "Throughput not available")
        er = metrics.get("error_rate")
        lines.append(f"Error rate: {er:.2%} ({metrics.get('error_count',0)} errors)" if er is not None else "Error rate not available")

    if anomalies.get("anomalies"):
        lines.append(f"Anomalies detected: {len(anomalies['anomalies'])} samples with unusual latency.")
    if anomalies.get("high_error_windows"):
        lines.append(f"High error windows detected: {len(anomalies['high_error_windows'])}")

    # Recommendations (simple rule-based)
    if metrics.get("p95_ms") and metrics["p95_ms"] > 2000:
        lines.append("Recommendation: p95 latency is > 2000 ms. Investigate slow endpoints and DB queries.")
    if metrics.get("error_rate") and metrics["error_rate"] > 0.05:
        lines.append("Recommendation: error rate > 5%. Check recent deploys and service logs.")

    return "\n".join(lines)


def generate_llm_summary_stub(context: Dict[str, Any]) -> str:
    """Placeholder function — replace with a real LLM call.

    Keep the context small (metrics, top slow endpoints, top anomalies) and send it to the LLM.
    Return the generated text.
    """
    # Example: pack a minimal JSON and instruct the LLM to produce a 3-sentence exec summary + 3 recommendations
    prompt = {
        "instructions": "You are a SRE. Produce a 3-sentence executive summary and 3 prioritized recommendations.",
        "context": context
    }
    # TODO: call your LLM SDK here (OpenAI, Local LLM, etc.) and return generated text.
    return "[LLM not configured] " + json.dumps(prompt)


# -------------------- Report orchestration --------------------

def create_report(path: str, keep: Optional[List[str]] = None, drop: Optional[List[str]] = None, use_llm: bool = False) -> Report:
    df = parse_jtl(path)
    df_clean = clean_dataframe(df, keep=keep, drop=drop)
    metrics = compute_metrics(df_clean)
    anomalies = detect_anomalies(df_clean)
    # prepare context for LLM (lightweight)
    context = {
        "metrics": metrics,
        "anomalies_count": len(anomalies.get("anomalies", [])),
        "top_slowest_labels": metrics.get("top_slowest_labels", [])[:5]
    }
    if use_llm:
        summary = generate_llm_summary_stub(context)
    else:
        summary = generate_text_summary(metrics, anomalies)

    report = Report(
        generated_at=datetime.utcnow().isoformat() + "Z",
        metrics=metrics,
        anomalies=anomalies,
        summary=summary,
        columns=list(df_clean.columns)
    )
    return report


def save_report(report: Report, out_path: str, fmt: str = "json") -> None:
    if fmt.lower() == "json":
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(report.to_json())
    elif fmt.lower() == "csv":
        # Flatten a small CSV-friendly representation: metrics + a summary row
        rows = [
            {"key": k, "value": json.dumps(v) if not isinstance(v, (str, int, float)) else v}
            for k, v in report.metrics.items()
        ]
        dfm = pd.DataFrame(rows)
        dfm.to_csv(out_path, index=False)
    else:
        raise ValueError("unsupported output format: use json or csv")


# -------------------- CLI --------------------

def _parse_csv_list(s: Optional[str]) -> Optional[List[str]]:
    if not s:
        return None
    return [x.strip() for x in s.split(",") if x.strip()]


def main():
    p = argparse.ArgumentParser(prog="jtl_reporter")
    p.add_argument("--input", "-i", required=True, help="Path to JTL (XML or CSV)")
    p.add_argument("--out", "-o", required=False, help="Path to write report (defaults to stdout for json)")
    p.add_argument("--format", "-f", default="json", choices=["json", "csv"], help="Report format")
    p.add_argument("--keep", help="Comma-separated list of columns to keep (optional)")
    p.add_argument("--drop", help="Comma-separated list of columns to drop (optional)")
    p.add_argument("--use-llm", action="store_true", help="Use LLM to generate summary (stub) if configured")

    args = p.parse_args()
    keep = _parse_csv_list(args.keep)
    drop = _parse_csv_list(args.drop)

    report = create_report(args.input, keep=keep, drop=drop, use_llm=args.use_llm)

    if args.out:
        save_report(report, args.out, fmt=args.format)
        print(f"Report written to {args.out}")
    else:
        print(report.to_json())


if __name__ == "__main__":
    main()
