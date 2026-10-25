#!/usr/bin/env python3
"""Run one calibration trial as a detached, lock-protected background process.

A single way to submit-and-wait for a trial so that multiple short-lived agent
invocations never launch competing submissions for the same one: it refuses to start a
second submission while one is in flight (a live PID in trial.lock), and a caller can
check that lock before spawning a new instance (squeue is blind to a submission still
retrying sbatch before any job reaches the queue).

Run from inside the agent's workspace:

    nohup python3 /path/to/src/run_trial_daemon.py \\
        --request next_trial.json \\
        --target ../../target/target_powerspectrum.csv \\
        --box-config ../../box_config/hr_box_config.py:HR_BOX \\
        > trial_daemon.log 2>&1 &
    disown

Writes alongside --request: trial.lock (this PID), trial.status (JSON state:
writing_paramfiles|submitting|running|done|failed), and trial.result.json (full
result, once done).
"""
import argparse
import atexit
import csv
import importlib.util
import json
import os
import shutil
import signal
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import calibration_env as env  # noqa: E402


def _load_box_config(spec: str):
    """spec is 'path/to/module.py:ATTR_NAME'."""
    path_str, attr = spec.rsplit(":", 1)
    path = Path(path_str).resolve()
    spec_obj = importlib.util.spec_from_file_location(path.stem, path)
    module = importlib.util.module_from_spec(spec_obj)
    spec_obj.loader.exec_module(module)
    return getattr(module, attr)


def _load_target(path: Path):
    with open(path) as f:
        rows = list(csv.DictReader(f))
    return (
        [float(r["k_h_per_Mpc"]) for r in rows],
        [float(r["P_k"]) for r in rows],
        [int(r["N_modes"]) for r in rows],
    )


def _write_status(status_path: Path, state: str, **extra) -> None:
    status_path.write_text(json.dumps({"state": state, **extra}, indent=2))


_LOCK_ACQUIRE_RETRIES = 5
_LOCK_READ_RETRIES = 5
_LOCK_READ_RETRY_DELAY_S = 0.05


def _acquire_lock(lock_path: Path) -> None:
    """Atomically claim trial.lock, self-healing a lock left by a dead process.

    os.O_CREAT | os.O_EXCL makes the create atomic, so only one process can win it;
    this closes a race where two daemons both see "no live lock" and both proceed onto
    the same trial_dir. The rest is diagnostics for whether to retry.
    """
    for _ in range(_LOCK_ACQUIRE_RETRIES):
        try:
            fd = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            os.write(fd, str(os.getpid()).encode())
            os.close(fd)
            return
        except FileExistsError:
            pass

        # A lock a live sibling just created can briefly read back empty (open and
        # write are separate syscalls), so retry an unparseable read before treating it
        # as stale.
        pid = None
        for _ in range(_LOCK_READ_RETRIES):
            try:
                pid = int(lock_path.read_text().strip())
                break
            except (ValueError, FileNotFoundError):
                time.sleep(_LOCK_READ_RETRY_DELAY_S)
        if pid is not None:
            try:
                os.kill(pid, 0)  # no-op signal: raises if pid is dead/not ours
                print(f"trial.lock names live PID {pid} -- refusing to start a duplicate "
                      "submission. Check trial.status instead of launching another.", file=sys.stderr)
                sys.exit(1)
            except (ProcessLookupError, PermissionError, OSError):
                pass  # dead -- fall through to clear and retry

        lock_path.unlink(missing_ok=True)  # stale -- clear and retry the atomic create

    print(f"could not acquire {lock_path} after {_LOCK_ACQUIRE_RETRIES} attempts "
          "-- persistent contention, giving up.", file=sys.stderr)
    sys.exit(1)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--request", required=True, type=Path)
    parser.add_argument("--target", required=True, type=Path)
    parser.add_argument("--box-config", required=True, help="path/to/module.py:ATTR")
    args = parser.parse_args()

    workspace = args.request.resolve().parent
    lock_path = workspace / "trial.lock"
    status_path = workspace / "trial.status"
    result_path = workspace / "trial.result.json"

    _acquire_lock(lock_path)

    def _cleanup() -> None:
        lock_path.unlink(missing_ok=True)

    def _on_signal(signum, _frame):
        _cleanup()
        sys.exit(1)

    atexit.register(_cleanup)
    signal.signal(signal.SIGTERM, _on_signal)
    signal.signal(signal.SIGINT, _on_signal)

    request = json.loads(args.request.read_text())
    trial_id = request["trial_id"]

    try:
        target_k, target_pk, target_nmodes = _load_target(args.target)
        box_config = _load_box_config(args.box_config)

        trial_dir = workspace / "trials" / f"trial_{trial_id}"
        if trial_dir.exists():
            # Leftover from a prior daemon that died before/during submission. Safe to
            # clear only if no job was ever submitted under this job_name; if one was, a
            # sibling daemon owns this trial, so leave its files alone.
            existing_job = env.job_exists_for_name(request["job_name"])
            if existing_job is not None:
                print(f"trials/trial_{trial_id} already has Slurm job {existing_job} under "
                      f"job_name={request['job_name']!r} -- a sibling daemon already owns this "
                      "trial. Not touching trial_dir or trial.status; exiting without resubmitting.",
                      file=sys.stderr)
                return
            shutil.rmtree(trial_dir)

        # run_trial() runs the write/submit/wait/score sequence; on_status mirrors its
        # stage transitions into trial.status. It never writes the terminal done/failed
        # states -- only this daemon does, since it also persists trial.result.json.
        result = env.run_trial(
            request["params"], trial_dir, target_k, target_pk, target_nmodes,
            job_name=request["job_name"], seed=request.get("seed"),
            sigma_mode=request.get("sigma_mode", "paired"), box_config=box_config,
            on_status=lambda state, extra: _write_status(status_path, state, trial_id=trial_id, **extra),
        )

        if not result.success:
            _write_status(status_path, "failed", trial_id=trial_id, job_id=result.job_id,
                          error=result.error, cpu_hours=result.cpu_hours, wall_time_min=result.wall_time_min)
            return

        result_path.write_text(json.dumps({
            "trial_id": trial_id, "params": request["params"], "seed": request.get("seed"),
            "job_id": result.job_id, "chi2": result.chi2, "reduced_chi2": result.reduced_chi2,
            "n_bins": result.n_bins, "table": result.table,
            "cpu_hours": result.cpu_hours, "wall_time_min": result.wall_time_min,
        }, indent=2))
        _write_status(status_path, "done", trial_id=trial_id, job_id=result.job_id,
                      chi2=result.chi2, reduced_chi2=result.reduced_chi2,
                      cpu_hours=result.cpu_hours, wall_time_min=result.wall_time_min)
    except Exception as e:
        _write_status(status_path, "failed", trial_id=trial_id, error=str(e))
        raise


if __name__ == "__main__":
    main()
