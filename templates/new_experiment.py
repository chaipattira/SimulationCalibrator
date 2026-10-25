#!/usr/bin/env python3
import argparse
import os
import shutil
import stat
import sys
from pathlib import Path

TEMPLATE_DIR = Path(__file__).resolve().parent
REPO_ROOT = TEMPLATE_DIR.parents[1]
EXPERIMENTS_DIR = REPO_ROOT / "experiments"
EXPERIMENTS_DATA_ROOT = Path("/work/hdd/bdne") / os.environ["USER"] / "SimulationCalibrator-experiments"

# Files in this template that are scaffolding for this script itself, not part of the
# experiment shape to copy into a new instance.
EXCLUDE = {"README.md", "new_experiment.py"}


def substitute(text: str, tokens: dict) -> str:
    for key, value in tokens.items():
        text = text.replace(key, value)
    return text


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--name", required=True,
                         help="experiments/<name> directory to create, e.g. my-new-run")
    parser.add_argument("--job-prefix", required=True,
                         help="short, unique slug for Slurm job names (e.g. 'exp1') -- "
                              "used as calib-<prefix>-llm- and calib-<prefix>-judge. Keep "
                              "it short and distinct from every other prefix used by runs "
                              "that might run concurrently, or their drivers' "
                              "wait_for_jobs_clear() squeue checks will cross-contaminate.")
    parser.add_argument("--seed", type=int, default=None,
                         help="RNG seed for the theta_fid draw (default: derived from --name, "
                              "for reproducibility without having to remember a number)")
    parser.add_argument("--skip", type=int, default=0,
                         help="draws to skip before taking theta_fid from the --seed stream "
                              "(0 unless you're extending an existing shared-seed theta batch "
                              "-- see judge/generate_theta_fid.py's docstring)")
    args = parser.parse_args()

    link_path = EXPERIMENTS_DIR / args.name
    dest = EXPERIMENTS_DATA_ROOT / args.name
    if link_path.exists() or link_path.is_symlink():
        sys.exit(f"error: {link_path} already exists -- pick a different --name or remove it first")
    if dest.exists():
        sys.exit(f"error: {dest} already exists on shared storage (but {link_path} doesn't) -- "
                  f"pick a different --name or remove the stale directory first")
    dest.mkdir(parents=True)

    seed = args.seed if args.seed is not None else (abs(hash(args.name)) % 100_000_000)

    tokens = {
        "__EXPERIMENT_NAME__": args.name,
        "__JOB_PREFIX__": args.job_prefix,
        "__SEED__": str(seed),
        "__SKIP__": str(args.skip),
        "__REPO_ROOT__": str(REPO_ROOT),
    }

    n_written = 0
    for src in sorted(TEMPLATE_DIR.rglob("*")):
        rel = src.relative_to(TEMPLATE_DIR)
        if rel.parts and rel.parts[0] in EXCLUDE:
            continue
        if src.is_dir():
            continue

        rel_dest = Path(*[p[:-5] if p.endswith(".tmpl") else p
                           for p in (rel.parts[:-1] + (rel.name[:-5] if rel.name.endswith(".tmpl") else rel.name,))])
        out_path = dest / rel_dest
        out_path.parent.mkdir(parents=True, exist_ok=True)

        if src.suffix == ".tmpl":
            text = substitute(src.read_text(), tokens)
            out_path.write_text(text)
        else:
            shutil.copyfile(src, out_path)
        n_written += 1

    (dest / "target").mkdir(parents=True, exist_ok=True)

    for driver in (dest / "code").glob("*.sh"):
        driver.chmod(driver.stat().st_mode | stat.S_IEXEC | stat.S_IXGRP | stat.S_IXOTH)

    # experiments/<name> is a symlink onto shared scratch storage (EXPERIMENTS_DATA_ROOT)
    # -- created last, only once dest is fully populated, so nothing can observe
    # experiments/<name> mid-write.
    link_path.symlink_to(dest)

    print(f"Wrote {n_written} files to {dest}")
    print(f"({link_path} -> {dest})")
    print(f"theta_fid draw: seed={seed}, skip={args.skip}")
    print("\nNext steps:")
    print(f"  1. cd {link_path.relative_to(REPO_ROOT)} && ../../.venv/bin/python3 judge/generate_theta_fid.py")
    print("  2. ../../.venv/bin/python3 judge/generate_target.py  (real Slurm job)")
    print("  3. Set budget_cpu_hours_cap (currently null) in agent_workspace/trial_state.yaml.")
    print("  4. Launch code/driver_llm-agent.sh inside tmux.")


if __name__ == "__main__":
    main()
