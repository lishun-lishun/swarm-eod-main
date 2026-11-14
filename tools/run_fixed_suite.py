"""Run the complete fixed-layout rescue suite in isolated subprocesses."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
from pathlib import Path
import subprocess
import sys
import time


ROOT = Path(__file__).resolve().parents[1]
TOOLS = ROOT / "tools"
DEFAULT_MAPS = [
    "Map01", "Map02", "Map03", "Map04", "Map05",
    "Map06", "Map07", "Map07t4", "Map08", "Map09",
]


def source_digest() -> str:
    digest = hashlib.sha256()
    files = [ROOT / "src" / "swarm_rescue" / "solutions" / "my_drone_eval.py"]
    files.extend(sorted((ROOT / "src" / "swarm_rescue" / "solutions" /
                         "blue_pid_explore").glob("*.py")))
    for path in files:
        digest.update(path.relative_to(ROOT).as_posix().encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--label", required=True)
    parser.add_argument("--engine-seed", type=int, default=861)
    parser.add_argument("--maps", nargs="*", default=DEFAULT_MAPS)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    suite_dir = ROOT / "reports" / "fixed_suites" / args.label
    suite_dir.mkdir(parents=True, exist_ok=True)
    digest = source_digest()
    started = time.time()
    records = []
    for map_name in args.maps:
        output = suite_dir / f"{map_name}_e{args.engine_seed}.json"
        if not (args.resume and output.exists()):
            command = [
                sys.executable,
                str(TOOLS / "benchmark_fixed_zones.py"),
                "--map", map_name,
                "--engine-seed", str(args.engine_seed),
                "--out", str(output),
            ]
            print(f"RUN {map_name}", flush=True)
            child_env = os.environ.copy()
            if (child_env.get("V5_DEADLINE_ASSIGNMENT") == "1"
                    and "V5_TIME_HORIZON" not in child_env):
                # The published map class declares the round limit. Supply
                # it through the independent test harness; the competition
                # controller never reads the map instance or hidden state.
                map_source = (ROOT / "src" / "swarm_rescue" / "maps"
                              / f"map_{map_name[3:].lower()}.py")
                match = re.search(r"_max_timestep_limit\s*=\s*(\d+)",
                                  map_source.read_text(encoding="utf-8"))
                if match is None:
                    raise ValueError(f"public time limit absent: {map_source}")
                child_env["V5_TIME_HORIZON"] = match.group(1)
            completed = subprocess.run(
                command, cwd=ROOT, env=child_env, check=False,
            )
            if completed.returncode != 0:
                records.append({"map": map_name, "error": completed.returncode})
                continue
        record = json.loads(output.read_text(encoding="utf-8"))
        records.append({
            "map": map_name,
            "disposed": record.get("disposed"),
            "total_bombs": record.get("total_bombs"),
            "official_total": record.get("official_total"),
            "mean_health_percent": record.get("mean_health_percent"),
            "time_to_dispose_all_step": record.get("time_to_dispose_all_step"),
            "elapsed_timestep": record.get("elapsed_timestep"),
            "max_timestep": record.get("max_timestep"),
            "walltime_seconds": record.get("walltime_seconds"),
            "path": str(output),
        })
        print(
            f"DONE {map_name}: {record.get('disposed')}/10 "
            f"score={record.get('official_total')}", flush=True,
        )

    valid = [row for row in records if row.get("official_total") is not None]
    summary = {
        "label": args.label,
        "engine_seed": args.engine_seed,
        "source_sha256": digest,
        "elapsed_wall_seconds": round(time.time() - started, 3),
        "maps": records,
        "full_maps": sum(row.get("disposed") == 10 for row in valid),
        "total_disposed": sum(int(row.get("disposed") or 0) for row in valid),
        "mean_official_score": (
            sum(float(row["official_total"]) for row in valid) / len(valid)
            if valid else None
        ),
        "mean_health_percent": (
            sum(float(row["mean_health_percent"]) for row in valid) / len(valid)
            if valid else None
        ),
    }
    summary_path = suite_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
