"""Test the exact official ZIP in an isolated official Launcher source tree."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import zipfile

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MAPS = ("Map01", "Map02", "Map03", "Map04", "Map05", "Map06",
                "Map07", "Map07t4", "Map08", "Map09")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--zip", type=Path, required=True)
    parser.add_argument("--label", required=True)
    parser.add_argument("--engine-seed", type=int, default=861)
    parser.add_argument("--maps", nargs="*", default=DEFAULT_MAPS)
    args = parser.parse_args()
    archive = args.zip.resolve()
    digest = hashlib.sha256(archive.read_bytes()).hexdigest()
    scratch = ROOT / "reports" / "v7_results"
    scratch.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="swarm_packaged_", dir=scratch) as name:
        temp = Path(name)
        package = temp / "src" / "swarm_rescue"
        shutil.copytree(
            ROOT / "src" / "swarm_rescue", package,
            ignore=lambda directory, children: [
                child for child in children
                if child == "__pycache__" or
                (Path(directory).resolve() ==
                 (ROOT / "src" / "swarm_rescue").resolve()
                 and child == "solutions")],
        )
        target = package / "solutions"
        target.mkdir()
        with zipfile.ZipFile(archive) as handle:
            for member in handle.infolist():
                path = (target / member.filename).resolve()
                if not path.is_relative_to(target.resolve()):
                    raise ValueError(f"unsafe archive member: {member.filename}")
            handle.extractall(target)
        (temp / "tools").mkdir()
        for tool in ("run_fixed_suite.py", "benchmark_fixed_zones.py",
                     "benchmark_competition.py"):
            shutil.copy2(ROOT / "tools" / tool, temp / "tools" / tool)
        (temp / "config").mkdir()
        shutil.copy2(ROOT / "config" / "zones_location.txt",
                     temp / "config" / "zones_location.txt")
        env = os.environ.copy()
        env["PYTHONPATH"] = str(temp / "src")
        env["V6_EVENT_AUDIT"] = "1"  # Independent evaluator only.
        for key in list(env):
            if key.startswith(("V4_", "V5_", "V6_", "V7_")) and key != "V6_EVENT_AUDIT":
                env.pop(key)
        command = [sys.executable, str(temp / "tools" / "run_fixed_suite.py"),
                   "--label", args.label, "--engine-seed", str(args.engine_seed),
                   "--maps", *args.maps]
        completed = subprocess.run(command, cwd=temp, env=env, check=False)
        source = temp / "reports" / "fixed_suites" / args.label
        destination = ROOT / "reports" / "fixed_suites" / args.label
        if not source.exists():
            raise RuntimeError(f"packaged test produced no output: {completed.returncode}")
        shutil.copytree(source, destination, dirs_exist_ok=True)
        summary_path = destination / "summary.json"
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        summary["official_submission_zip"] = str(archive)
        summary["official_submission_zip_sha256"] = digest
        summary["isolated_installed_zip_code"] = True
        for row in summary["maps"]:
            if "path" in row:
                row["path"] = str(destination / Path(row["path"]).name)
        summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        if completed.returncode:
            raise SystemExit(completed.returncode)
        print(f"ZIP SHA256 {digest}")
        print(f"Results: {summary_path}")


if __name__ == "__main__":
    main()
