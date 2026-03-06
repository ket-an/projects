import subprocess
import os
import re
from flask import Flask, render_template, jsonify, request

app = Flask(__name__)

state = {
    "repo_url": None,
    "local_path": None,
    "cloned": False,
    "workload_name": None
}


def run_git(args: list, cwd: str = None) -> dict:
    try:
        result = subprocess.run(
            ["git"] + args,
            cwd=cwd or state["local_path"],
            capture_output=True,
            text=True
        )
        return {
            "stdout": result.stdout.strip(),
            "stderr": result.stderr.strip(),
            "returncode": result.returncode,
            "success": result.returncode == 0
        }
    except FileNotFoundError:
        return {"stdout": "", "stderr": "Git not found in PATH.", "returncode": -1, "success": False}
    except Exception as e:
        return {"stdout": "", "stderr": str(e), "returncode": -1, "success": False}


def parse_porcelain_line(line: str):
    """
    Standard porcelain v1: 'XY filename' — X=col0, Y=col1, space=col2, filename=col3+
    But some git versions / configs emit a short form: 'X filename' — only 1 status char.
    We detect which format is in use by checking col1:
      - If col1 is a space AND col2 is also a space → short form, filename starts at col2
      - Otherwise → standard form, filename starts at col3
    """
    if len(line) < 3:
        return None

    # Detect format by checking if col1 is the separator space
    # Standard: 'XY filename' → col2 is space, filename at [3:]
    # Short:    'X  filename' or 'X filename' → col1 is space, filename at [2:]
    if line[1] == ' ':
        # Short format — only one status character
        X        = line[0]
        Y        = ' '
        filename = line[2:]
    else:
        # Standard format — two status characters
        X        = line[0]
        Y        = line[1]
        filename = line[3:]

    # Handle renamed: "old -> new"
    if " -> " in filename:
        filename = filename.split(" -> ")[-1]

    # Strip surrounding quotes git adds for special chars
    filename = filename.strip()
    if filename.startswith('"') and filename.endswith('"'):
        filename = filename[1:-1]

    return X + Y, filename


@app.route("/")
def index():
    return render_template("index.html")


# ── Setup ─────────────────────────────────────────────────────────────────────

@app.route("/api/setup", methods=["POST"])
def setup():
    data          = request.get_json(silent=True) or {}
    repo_url      = data.get("repo_url", "").strip()
    local_path    = data.get("local_path", "").strip()
    workload_name = data.get("workload_name", "").strip()

    if not repo_url or not local_path:
        return jsonify({"success": False, "error": "Repo URL and local path are required."}), 400

    local_path = os.path.expanduser(local_path)
    state.update({
        "repo_url":      repo_url,
        "local_path":    local_path,
        "cloned":        False,
        "workload_name": workload_name or None
    })

    git_dir = os.path.join(local_path, ".git")
    if os.path.exists(git_dir):
        state["cloned"] = True
        remote = run_git(["remote", "get-url", "origin"])
        return jsonify({
            "success": True,
            "message": f"Repo already exists at '{local_path}'.",
            "log": f"Remote origin: {remote['stdout']}",
            "error": remote["stderr"],
            "already_cloned": True
        })

    os.makedirs(local_path, exist_ok=True)
    result = subprocess.run(
        ["git", "clone", repo_url, local_path],
        capture_output=True, text=True
    )
    state["cloned"] = result.returncode == 0

    return jsonify({
        "success":        result.returncode == 0,
        "message":        f"Cloned '{repo_url}' → '{local_path}'" if result.returncode == 0 else "Clone failed.",
        "log":            result.stdout.strip(),
        "error":          result.stderr.strip(),
        "already_cloned": False
    })


@app.route("/api/config")
def get_config():
    return jsonify({
        "repo_url":      state["repo_url"],
        "local_path":    state["local_path"],
        "cloned":        state["cloned"],
        "workload_name": state["workload_name"]
    })


# ── Directory Validation ───────────────────────────────────────────────────────

@app.route("/api/validate")
def validate_structure():
    if not state["cloned"]:
        return jsonify({"success": False, "error": "No repo configured.", "checks": []})

    root          = state["local_path"]
    workload_name = state["workload_name"]
    checks        = []
    all_passed    = True

    def add_check(name, passed, message):
        nonlocal all_passed
        if not passed:
            all_passed = False
        checks.append({"name": name, "passed": passed, "message": message})

    # ── bin/ directory ──────────────────────────────────────────────────────
    bin_path = os.path.join(root, "bin")
    if not os.path.isdir(bin_path):
        add_check("bin/ exists", False, "'bin/' directory not found in repo root.")
    else:
        bin_files = os.listdir(bin_path)

        jmx_files  = [f for f in bin_files if f.endswith(".jmx")]
        prop_files = [f for f in bin_files if f.endswith(".properties")]

        # Exactly one .jmx
        if len(jmx_files) == 0:
            add_check("bin/ has 1 .jmx", False, "No .jmx file found in bin/.")
        elif len(jmx_files) > 1:
            add_check("bin/ has 1 .jmx", False, f"Multiple .jmx files found in bin/: {', '.join(jmx_files)}")
        else:
            add_check("bin/ has 1 .jmx", True, f"Found: {jmx_files[0]}")

        # Exactly one .properties
        if len(prop_files) == 0:
            add_check("bin/ has 1 .properties", False, "No .properties file found in bin/.")
        elif len(prop_files) > 1:
            add_check("bin/ has 1 .properties", False, f"Multiple .properties files found: {', '.join(prop_files)}")
        else:
            add_check("bin/ has 1 .properties", True, f"Found: {prop_files[0]}")

        # Workload name check (only if workload_name provided)
        if workload_name:
            # .jmx name check
            if len(jmx_files) == 1:
                jmx_base = os.path.splitext(jmx_files[0])[0]
                jmx_ok   = workload_name.lower() in jmx_base.lower()
                add_check(
                    f".jmx name matches workload '{workload_name}'",
                    jmx_ok,
                    f"'{jmx_files[0]}' {'✔ contains' if jmx_ok else '✘ does not contain'} workload name '{workload_name}'."
                )

            # .properties name check
            if len(prop_files) == 1:
                prop_base = os.path.splitext(prop_files[0])[0]
                prop_ok   = workload_name.lower() in prop_base.lower()
                add_check(
                    f".properties name matches workload '{workload_name}'",
                    prop_ok,
                    f"'{prop_files[0]}' {'✔ contains' if prop_ok else '✘ does not contain'} workload name '{workload_name}'."
                )

    # ── data/ directory ─────────────────────────────────────────────────────
    data_path = os.path.join(root, "data")
    if not os.path.isdir(data_path):
        add_check("data/ exists", False, "'data/' directory not found in repo root.")
    else:
        csv_files = [f for f in os.listdir(data_path) if f.endswith(".csv")]
        if len(csv_files) == 0:
            add_check("data/ has ≥1 .csv", False, "No .csv files found in data/.")
        else:
            add_check("data/ has ≥1 .csv", True, f"Found {len(csv_files)} .csv file(s): {', '.join(csv_files)}")

    return jsonify({
        "success":  all_passed,
        "checks":   checks,
        "workload": workload_name or "(not set)"
    })


# ── Git Status ─────────────────────────────────────────────────────────────────

@app.route("/api/status")
def git_status():
    if not state["cloned"]:
        return jsonify({"success": False, "error": "No repo configured.", "files": []})

    result = run_git(["status", "--porcelain"])
    files  = []
    status_map = {
        "M": "Modified", "A": "Added", "D": "Deleted",
        "R": "Renamed",  "C": "Copied","U": "Unmerged",
        "?": "Untracked"," ": "Unchanged"
    }

    if result["success"] and result["stdout"]:
        for line in result["stdout"].splitlines():
            parsed = parse_porcelain_line(line)
            if not parsed:
                continue
            xy, filename = parsed
            files.append({
                "filename":     filename,
                "index_status": status_map.get(xy[0], xy[0]),
                "work_status":  status_map.get(xy[1], xy[1]),
                "raw":          xy
            })

    return jsonify({
        "files":   files,
        "log":     result["stdout"] or "(No changes detected)",
        "error":   result["stderr"],
        "success": result["success"]
    })


@app.route("/api/gitstatus")
def git_full_status():
    """Return full `git status` output and raw porcelain for debugging."""
    if not state["cloned"]:
        return jsonify({"success": False, "error": "No repo configured.", "full": "", "raw": ""})

    full   = run_git(["status"])
    raw    = run_git(["status", "--porcelain"])
    return jsonify({
        "full":    full["stdout"],
        "raw":     raw["stdout"],
        "error":   full["stderr"] or raw["stderr"],
        "success": full["success"]
    })


@app.route("/api/debug/rawstatus")
def debug_raw_status():
    """Debug: return raw bytes representation of porcelain output to diagnose parsing issues."""
    if not state["cloned"]:
        return jsonify({"success": False, "error": "No repo configured."})
    result = run_git(["status", "--porcelain"])
    lines  = []
    for line in result["stdout"].splitlines():
        parsed   = parse_porcelain_line(line)
        xy, fname = parsed if parsed else ("??", "PARSE ERROR")
        lines.append({
            "raw_line":       line,
            "repr":           repr(line),
            "len":            len(line),
            "col0":           repr(line[0]) if len(line) > 0 else "",
            "col1":           repr(line[1]) if len(line) > 1 else "",
            "col2":           repr(line[2]) if len(line) > 2 else "",
            "xy_parsed":      xy,
            "filename_parsed": fname,
        })
    return jsonify({"lines": lines, "success": result["success"], "stderr": result["stderr"]})


# ── Git Add ────────────────────────────────────────────────────────────────────

@app.route("/api/add", methods=["POST"])
def git_add():
    if not state["cloned"]:
        return jsonify({"success": False, "error": "No repo configured."})

    data  = request.get_json(silent=True) or {}
    files = data.get("files", [])

    # Pass filenames exactly as git reported them (forward slashes, relative to repo root)
    # Do NOT use os.path.join — it corrupts forward-slash paths on Windows
    if files:
        result = run_git(["add", "--"] + files)
        label  = ", ".join(files)
    else:
        result = run_git(["add", "-A"])
        label  = "All files"

    return jsonify({
        "message": f"Staged: {label}",
        "log":     result["stdout"] or "Files staged successfully.",
        "error":   result["stderr"],
        "success": result["success"]
    })


# ── Git Commit ─────────────────────────────────────────────────────────────────

@app.route("/api/commit", methods=["POST"])
def git_commit():
    if not state["cloned"]:
        return jsonify({"success": False, "error": "No repo configured."})
    data = request.get_json(silent=True) or {}
    msg  = data.get("message", "").strip()
    if not msg:
        return jsonify({"success": False, "error": "Commit message is required."}), 400
    result = run_git(["commit", "-m", msg])
    return jsonify({
        "message": f"Committed: {msg}",
        "log":     result["stdout"],
        "error":   result["stderr"],
        "success": result["success"]
    })


# ── Git Push ───────────────────────────────────────────────────────────────────

@app.route("/api/push", methods=["POST"])
def git_push():
    if not state["cloned"]:
        return jsonify({"success": False, "error": "No repo configured."})
    br     = run_git(["rev-parse", "--abbrev-ref", "HEAD"])
    branch = br["stdout"] if br["success"] else "main"
    result = run_git(["push", "origin", branch])
    return jsonify({
        "message": f"Pushed to origin/{branch}",
        "log":     result["stdout"],
        "error":   result["stderr"],
        "success": result["success"]
    })


# ── Git Log / Branch ───────────────────────────────────────────────────────────

@app.route("/api/log")
def git_log():
    if not state["cloned"]:
        return jsonify({"success": False, "log": "", "error": "No repo configured."})
    result = run_git(["log", "--oneline", "-10"])
    return jsonify({"log": result["stdout"] or "(No commits yet)", "error": result["stderr"], "success": result["success"]})


@app.route("/api/branch")
def git_branch():
    if not state["cloned"]:
        return jsonify({"branch": "—", "success": False})
    result = run_git(["rev-parse", "--abbrev-ref", "HEAD"])
    return jsonify({"branch": result["stdout"] if result["success"] else "unknown", "error": result["stderr"], "success": result["success"]})


if __name__ == "__main__":
    print("[Git Agent] Starting on http://localhost:5000")
    app.run(debug=True, port=5000)