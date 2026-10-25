"""Shared calibration environment for MP-Gadget parameter-recovery runs.

Every method under test (LLM agent, random search, GP+BO) calls run_trial() with the
same parameters and gets back the same kind of result, computed the same way, so that
comparisons between methods are about the proposal strategy, not the mechanics.

Default box: MP-Gadget's examples/star-small (BoxConfig below selects others). Target
statistic: matter power spectrum P(k) at the final snapshot, scored with a
mode-count-weighted chi-squared.
"""
from __future__ import annotations

import math
import os
import re
import subprocess
import sys
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

TUNABLE_PARAMS = [
    "Omega0",
    "OmegaBaryon",
    "Sigma8",
    "WindSigma0",
    "WindSpeedFactor",
    "BlackHoleFeedbackFactor",
]
# WindSigma0 (km/s) controls mass loading and is live under the ofjt10 wind model;
# WindEnergyFraction only feeds an inactive code path there, so it is not tuned.

# Optional parameters, verified live under star-small's compiled defaults. Not in
# TUNABLE_PARAMS: applied only if present in params, else left at the template default,
# so six-parameter callers are unaffected. Excluded on purpose: PrimordialAmp
# (degenerate with Sigma8, which GenIC already rescales to), WindEfficiency (dead under
# ofjt10), Ngrid/BoxSize (resolution, not physics -- would change the FFT binning and
# break comparability with the target).
OPTIONAL_TUNABLE_PARAMS = [
    "PrimordialIndex",           # genic: spectral tilt, feeds CLASS via make_class_power.py
    "HubbleParam",                # genic: feeds CLASS cosmology + gadget inherits from IC header
    "WindFreeTravelLength",        # gadget: live under ofjt10 (WIND_DECOUPLE_SPH)
    "WindFreeTravelDensFac",       # gadget: live under ofjt10 (WIND_DECOUPLE_SPH)
    "BlackHoleAccretionFactor",    # gadget: unconditional, feeds Bondi accretion rate
    "SeedBlackHoleMass",           # gadget: unconditional, feeds BH seed mass
]

MPGADGET_ROOT = Path(os.environ.get("MPGADGET_ROOT", "/u/ctirapongprasert/SimulationCalibrator/sims/MP-Gadget"))
STAR_SMALL_DIR = MPGADGET_ROOT / "examples" / "star-small"
FINAL_SCALEFACTOR = "0.2000"  # star-small OutputList's last entry, %.4f-formatted filename suffix

sys.path.insert(0, str(MPGADGET_ROOT / "tools"))
from make_class_power import make_class_power  # noqa: E402
# Regenerate the CLASS power spectrum per trial: a fixed precomputed table is only
# exact at the cosmology it was made with, so varying Omega0/OmegaBaryon would reweight
# frozen transfer-function shapes rather than recompute them.

SLURM_ACCOUNT = "bdne-delta-cpu"
SLURM_PARTITION = "cpu-interactive"
NTASKS = 2
CPUS_PER_TASK = 2
MEM = "8G"
JOB_TIME = "00:30:00"

# sbatch retry: a QOS per-user job cap can reject submission while other jobs occupy
# the cap -- transient contention, not a failure of this trial. Handled here so every
# submit_job() path is covered; otherwise an unretried rejection looks like a failed
# trial.
SUBMIT_MAX_RETRIES = 60
SUBMIT_RETRY_BACKOFF_S = 60

# Watchdog: MP-Gadget can hang (some MPI ranks abort, others block in a collective)
# rather than exit cleanly. Poll the job's stdout log for growth; if it stalls this
# long, scancel it.
STALL_TIMEOUT_S = 6 * 60
POLL_INTERVAL_S = 10

# k-range trim: drop bins with too few modes -- they are sample-variance dominated at
# the fundamental-mode end.
MIN_MODES = 20

# High-k cut: past the particle-grid Nyquist, shot noise/discreteness dominates the
# true signal (well before the PM mesh aliases), so those bins are resolution
# artifacts, not physical information -- drop them regardless of mode count.
IC_NGRID = 64            # examples/star-small paramfile.genic Ngrid
BOX_SIZE_MPC_H = 5.0      # examples/star-small paramfile.genic BoxSize, kpc/h -> Mpc/h
K_MAX_H_PER_MPC = math.pi * IC_NGRID / BOX_SIZE_MPC_H  # ~= 40.21 h/Mpc


@dataclass(frozen=True)
class BoxConfig:
    """What differs between calibration boxes: the paramfile template, the k-range for
    the chi2 evaluation, and the Slurm resources a trial needs. Call sites default to
    STAR_SMALL_BOX; pass a different BoxConfig for another box.
    """
    name: str
    template_dir: Path
    box_size_mpc_h: float
    ic_ngrid: int
    final_scalefactor: str  # %.4f-formatted suffix of output/powerspectrum-<a>.txt
    min_modes: int
    k_max_h_per_mpc: float
    k_min_h_per_mpc: float = 0.0
    partition: str = SLURM_PARTITION
    ntasks: int = NTASKS
    cpus_per_task: int = CPUS_PER_TASK
    mem: str = MEM
    job_time: str = JOB_TIME
    stall_timeout_s: int = STALL_TIMEOUT_S


STAR_SMALL_BOX = BoxConfig(
    name="star-small",
    template_dir=STAR_SMALL_DIR,
    box_size_mpc_h=BOX_SIZE_MPC_H,
    ic_ngrid=IC_NGRID,
    final_scalefactor=FINAL_SCALEFACTOR,
    min_modes=MIN_MODES,
    k_max_h_per_mpc=K_MAX_H_PER_MPC,
)


@dataclass
class TrialResult:
    success: bool
    params: dict
    chi2: float | None = None
    reduced_chi2: float | None = None
    n_bins: int | None = None
    table: list | None = None  # list of dicts: k, P_target, P_sim, frac_resid, chi2_i
    cpu_hours: float = 0.0
    wall_time_min: float = 0.0
    job_id: int | None = None
    error: str | None = None


def _set_param(lines: list[str], name: str, value) -> list[str]:
    """Replace a param line if present, else append it. Matches 'Name = ...' or 'Name=...'."""
    pattern = re.compile(rf"^\s*{re.escape(name)}\s*=")
    new_line = f"{name} = {value}\n"
    out = []
    replaced = False
    for line in lines:
        if pattern.match(line):
            out.append(new_line)
            replaced = True
        else:
            out.append(line)
    if not replaced:
        out.append(new_line)
    return out


def write_paramfiles(
    params: dict, rundir: Path, seed: int | None = None, box_config: BoxConfig = STAR_SMALL_BOX,
) -> None:
    """Write paramfile.genic and paramfile.gadget for this trial into rundir.

    seed: if given, overrides the IC Seed for this trial (draw an independent
    realization per trial); default None leaves the template's Seed untouched
    (phase-paired design).

    box_config: which paramfile template to start from and which box constants to use
    (default STAR_SMALL_BOX).
    """
    missing = set(TUNABLE_PARAMS) - set(params)
    if missing:
        raise ValueError(f"missing required params: {missing}")
    rundir.mkdir(parents=True, exist_ok=True)

    omega_lambda = 1.0 - params["Omega0"]  # flat LCDM: not a free parameter here

    # Regenerate CLASS for this trial's Omega0/OmegaBaryon (Sigma8 is a pure amplitude
    # rescale GenIC applies separately). make_class_power() won't overwrite -- fine,
    # since rundir is fresh per trial.
    genic_lines = (box_config.template_dir / "paramfile.genic").read_text().splitlines(keepends=True)
    genic_lines = _set_param(genic_lines, "Omega0", params["Omega0"])
    genic_lines = _set_param(genic_lines, "OmegaLambda", omega_lambda)
    genic_lines = _set_param(genic_lines, "OmegaBaryon", params["OmegaBaryon"])
    genic_lines = _set_param(genic_lines, "Sigma8", params["Sigma8"])
    if seed is not None:
        genic_lines = _set_param(genic_lines, "Seed", seed)
    # Optional params: applied only if present. HubbleParam in genic suffices -- gadget
    # inherits it from the IC header when its own paramfile omits it, as here.
    if "PrimordialIndex" in params:
        genic_lines = _set_param(genic_lines, "PrimordialIndex", params["PrimordialIndex"])
    if "HubbleParam" in params:
        genic_lines = _set_param(genic_lines, "HubbleParam", params["HubbleParam"])
    # Bare filename, not rundir-prefixed: make_class_power() rejoins this against the
    # paramfile's own directory, and GenIC runs with cwd=rundir (see SBATCH_TEMPLATE).
    genic_lines = _set_param(genic_lines, "FileWithInputSpectrum", "class_pk.dat")
    genic_lines = _set_param(genic_lines, "DifferentTransferFunctions", 0)
    (rundir / "paramfile.genic").write_text("".join(genic_lines))
    make_class_power(str(rundir / "paramfile.genic"))

    gadget_lines = (box_config.template_dir / "paramfile.gadget").read_text().splitlines(keepends=True)
    gadget_lines = _set_param(gadget_lines, "Omega0", params["Omega0"])
    gadget_lines = _set_param(gadget_lines, "OmegaLambda", omega_lambda)
    gadget_lines = _set_param(gadget_lines, "OmegaBaryon", params["OmegaBaryon"])
    gadget_lines = _set_param(gadget_lines, "WindSigma0", params["WindSigma0"])
    gadget_lines = _set_param(gadget_lines, "WindSpeedFactor", params["WindSpeedFactor"])
    gadget_lines = _set_param(gadget_lines, "BlackHoleFeedbackFactor", params["BlackHoleFeedbackFactor"])
    # Optional params -- see the note above the genic block.
    if "WindFreeTravelLength" in params:
        gadget_lines = _set_param(gadget_lines, "WindFreeTravelLength", params["WindFreeTravelLength"])
    if "WindFreeTravelDensFac" in params:
        gadget_lines = _set_param(gadget_lines, "WindFreeTravelDensFac", params["WindFreeTravelDensFac"])
    if "BlackHoleAccretionFactor" in params:
        gadget_lines = _set_param(gadget_lines, "BlackHoleAccretionFactor", params["BlackHoleAccretionFactor"])
    if "SeedBlackHoleMass" in params:
        gadget_lines = _set_param(gadget_lines, "SeedBlackHoleMass", params["SeedBlackHoleMass"])
    (rundir / "paramfile.gadget").write_text("".join(gadget_lines))

    (rundir / "IC").mkdir(exist_ok=True)
    (rundir / "output").mkdir(exist_ok=True)


SBATCH_TEMPLATE = """#!/bin/bash
#SBATCH --job-name={job_name}
#SBATCH --account={account}
#SBATCH --partition={partition}
#SBATCH --time={time}
#SBATCH --nodes=1
#SBATCH --ntasks={ntasks}
#SBATCH --cpus-per-task={cpus_per_task}
#SBATCH --mem={mem}
#SBATCH --output=slurm.out
#SBATCH --error=slurm.err

set -euo pipefail
# Module versions pinned to ones known to load on the target cluster; a mismatch
# makes the job die immediately on "unknown module(s)" before GenIC/Gadget ever run.
module load gsl/2.8-gcc13.3.1 cray-fftw/3.3.10.11 cray-hdf5-parallel/1.14.3.9
export LD_LIBRARY_PATH="${{GSL_ROOT_DIR}}/lib:${{LD_LIBRARY_PATH:-}}"
export OMP_NUM_THREADS={cpus_per_task}

# No `cd` needed: sbatch runs with cwd=rundir (see submit_job()), so a `cd` here
# would double the relative path and make the job fail immediately.
srun -n {ntasks} "{mpgadget_root}/genic/MP-GenIC" paramfile.genic
srun -n {ntasks} "{mpgadget_root}/gadget/MP-Gadget" paramfile.gadget
"""


def _render_sbatch_script(job_name: str, box_config: BoxConfig = STAR_SMALL_BOX) -> str:
    return SBATCH_TEMPLATE.format(
        job_name=job_name,
        account=SLURM_ACCOUNT,
        partition=box_config.partition,
        time=box_config.job_time,
        ntasks=box_config.ntasks,
        cpus_per_task=box_config.cpus_per_task,
        mem=box_config.mem,
        mpgadget_root=MPGADGET_ROOT,
    )


def submit_job(rundir: Path, job_name: str, box_config: BoxConfig = STAR_SMALL_BOX) -> int:
    script = _render_sbatch_script(job_name, box_config)
    script_path = rundir / "submit.sbatch"
    script_path.write_text(script)
    # sbatch's argument resolves relative to cwd=rundir, so it must be a bare filename.
    last_error = None
    for attempt in range(1, SUBMIT_MAX_RETRIES + 1):
        try:
            out = subprocess.run(
                ["sbatch", "--parsable", script_path.name],
                cwd=rundir, capture_output=True, text=True, check=True,
            )
            return int(out.stdout.strip().split(";")[0])
        except subprocess.CalledProcessError as e:
            last_error = e
            if attempt == SUBMIT_MAX_RETRIES:
                raise
            time.sleep(SUBMIT_RETRY_BACKOFF_S)
    raise last_error


def _job_state(job_id: int) -> str | None:
    out = subprocess.run(
        ["squeue", "-j", str(job_id), "-h", "-o", "%T"],
        capture_output=True, text=True,
    )
    state = out.stdout.strip()
    return state or None  # None means it has left the queue


def job_exists_for_name(job_name: str) -> int | None:
    """Job id if a Slurm job with this name currently exists or ever ran, else None.

    Lets a caller tell a resubmit-safe stale trial_dir (no job was ever created) apart
    from an already-submitted trial whose files must not be touched. Checks squeue
    first (cheap, always current), then falls back to sacct for jobs that already
    finished and left the queue.
    """
    out = subprocess.run(
        ["squeue", "-n", job_name, "-h", "-o", "%A"],
        capture_output=True, text=True,
    )
    job_ids = out.stdout.split()
    if job_ids:
        return int(job_ids[0])

    out = subprocess.run(
        ["sacct", "--name", job_name, "-n", "-X", "-o", "JobID", "-S", "1970-01-01"],
        capture_output=True, text=True,
    )
    job_ids = out.stdout.split()
    return int(job_ids[0]) if job_ids else None


# States sacct can still report for a few seconds after a job has left squeue -- its
# accounting DB lags squeue. Retry until a genuine terminal state appears, so a stale
# non-terminal snapshot isn't mistaken for a failure that discards a real result.
_SACCT_NONTERMINAL_STATES = {"PENDING", "RUNNING", "CONFIGURING", "COMPLETING", "SUSPENDED", "REQUEUED"}
_SACCT_RETRY_S = 5
_SACCT_MAX_RETRIES = 6


def wait_for_job(job_id: int, rundir: Path, stall_timeout_s: int = STALL_TIMEOUT_S) -> dict:
    """Poll until the job leaves the queue, or scancel it if its log stalls.

    Returns dict with keys: state ('COMPLETED'|'FAILED'|'HUNG'|'TIMEOUT'), cpu_hours, wall_time_min.
    """
    log_path = rundir / "slurm.out"
    start = time.time()
    last_size = -1
    last_change = start

    while True:
        state = _job_state(job_id)
        if state is None:
            break  # left the queue
        if state == "PENDING":
            # Still queued, no node yet: slurm.out doesn't exist, so the stall clock
            # would trip on pure queue contention. Only a RUNNING job whose log isn't
            # growing is genuinely hung, so hold (and reset) the timer until it runs.
            last_size = -1
            last_change = time.time()
            time.sleep(POLL_INTERVAL_S)
            continue
        size = log_path.stat().st_size if log_path.exists() else 0
        now = time.time()
        if size != last_size:
            last_size = size
            last_change = now
        elif now - last_change > stall_timeout_s:
            subprocess.run(["scancel", str(job_id)], check=False)
            wall_min = (now - start) / 60.0
            return {"state": "HUNG", "cpu_hours": _sacct_cpu_hours(job_id), "wall_time_min": wall_min}
        time.sleep(POLL_INTERVAL_S)

    sacct_state = _sacct_final_state(job_id)
    retries = 0
    while sacct_state in _SACCT_NONTERMINAL_STATES and retries < _SACCT_MAX_RETRIES:
        time.sleep(_SACCT_RETRY_S)
        sacct_state = _sacct_final_state(job_id)
        retries += 1

    wall_min = (time.time() - start) / 60.0
    return {"state": sacct_state, "cpu_hours": _sacct_cpu_hours(job_id), "wall_time_min": wall_min}


def _sacct_final_state(job_id: int) -> str:
    out = subprocess.run(
        ["sacct", "-j", str(job_id), "-n", "-X", "-o", "State"],
        capture_output=True, text=True,
    )
    state = out.stdout.strip().splitlines()[0].strip() if out.stdout.strip() else "UNKNOWN"
    return state.split()[0]


def _sacct_cpu_hours(job_id: int) -> float:
    """CPUTimeRAW is in CPU-seconds, summed correctly only for the parent job step."""
    out = subprocess.run(
        ["sacct", "-j", str(job_id), "-n", "-X", "-o", "CPUTimeRAW"],
        capture_output=True, text=True,
    )
    try:
        cpu_seconds = int(out.stdout.strip().splitlines()[0].strip())
    except (ValueError, IndexError):
        return 0.0
    return cpu_seconds / 3600.0


def parse_powerspectrum(rundir: Path, scalefactor: str = FINAL_SCALEFACTOR) -> tuple[list, list, list]:
    path = rundir / "output" / f"powerspectrum-{scalefactor}.txt"
    if not path.exists():
        raise FileNotFoundError(f"no power spectrum output at {path}")
    k, pk, nmodes = [], [], []
    for line in path.read_text().splitlines():
        if line.startswith("#") or not line.strip():
            continue
        parts = line.split()
        k.append(float(parts[0]))
        pk.append(float(parts[1]))
        nmodes.append(int(float(parts[2])))
    return k, pk, nmodes


def compute_chi2(
    k_sim: list, pk_sim: list, nmodes_sim: list,
    k_target: list, pk_target: list, nmodes_target: list,
    min_modes: int = MIN_MODES,
    k_max: float = K_MAX_H_PER_MPC,
    k_min: float = 0.0,
    sigma_mode: str = "paired",
) -> tuple[float, float, list]:
    """Mode-count-weighted chi-squared. Assumes k_sim and k_target share the same
    binning (same box/grid across all trials and the target), true as long as only
    physics parameters vary, never resolution or box size.

    k_min: drop bins below this k (default 0.0 -- no low-k cut, relying on min_modes
    alone to trim the noisy fundamental-mode end). Set it for a larger box whose
    k-range is bounded on both ends.

    sigma_mode="paired" (default): sigma_i^2 = (2/N_target_i)*P_target_i^2 -- the
    variance of one noisy realization against an assumed-noiseless prediction. Correct
    when sim and target share the same IC Seed, so the sim's own realization noise
    cancels against the target's in the residual.

    sigma_mode="unpaired": also adds (2/N_sim_i)*P_sim_i^2 -- the correct denominator
    when sim and target use different (unpaired) IC seeds, so both sides carry
    independent realization noise.
    """
    if sigma_mode not in ("paired", "unpaired"):
        raise ValueError(f"sigma_mode must be 'paired' or 'unpaired', got {sigma_mode!r}")
    if len(k_sim) != len(k_target):
        raise ValueError(f"bin count mismatch: sim has {len(k_sim)}, target has {len(k_target)}")

    chi2 = 0.0
    table = []
    n_used = 0
    for i in range(len(k_target)):
        n = nmodes_target[i]
        if n < min_modes:
            continue
        if k_target[i] > k_max:
            continue
        if k_target[i] < k_min:
            continue
        sigma2 = (2.0 / n) * pk_target[i] ** 2
        if sigma_mode == "unpaired":
            sigma2 += (2.0 / nmodes_sim[i]) * pk_sim[i] ** 2
        resid = pk_sim[i] - pk_target[i]
        chi2_i = resid ** 2 / sigma2
        chi2 += chi2_i
        n_used += 1
        table.append({
            "k": k_target[i],
            "P_target": pk_target[i],
            "P_sim": pk_sim[i],
            "frac_resid": resid / pk_target[i],
            "chi2_i": chi2_i,
            "n_modes": n,
        })
    reduced_chi2 = chi2 / n_used if n_used > 0 else math.inf
    return chi2, reduced_chi2, table


def chi2_from_table(table: list, sigma_mode: str = "paired") -> tuple[float, float]:
    """Recompute (chi2, reduced_chi2) under a different sigma_mode from an already
    computed compute_chi2() table, without re-parsing or re-filtering bins. n_modes
    serves as both sides' mode count, since sim and target share the same binning."""
    if sigma_mode not in ("paired", "unpaired"):
        raise ValueError(f"sigma_mode must be 'paired' or 'unpaired', got {sigma_mode!r}")
    chi2 = 0.0
    for row in table:
        n = row["n_modes"]
        sigma2 = (2.0 / n) * row["P_target"] ** 2
        if sigma_mode == "unpaired":
            sigma2 += (2.0 / n) * row["P_sim"] ** 2
        resid = row["P_sim"] - row["P_target"]
        chi2 += resid ** 2 / sigma2
    reduced_chi2 = chi2 / len(table) if table else math.inf
    return chi2, reduced_chi2


def run_trial(
    params: dict, trial_dir: Path, target_k: list, target_pk: list, target_nmodes: list,
    job_name: str = "calib-trial",
    seed: int | None = None,
    sigma_mode: str = "paired",
    box_config: BoxConfig = STAR_SMALL_BOX,
    on_status: Callable[[str, dict], None] | None = None,
) -> TrialResult:
    """Run one trial end-to-end: write paramfiles, submit, wait, score against target.

    on_status: optional callback invoked as on_status(state, extra) right before each
    stage ("writing_paramfiles", "submitting", then "running" with
    extra={"job_id": ...}), for a caller that mirrors trial progress elsewhere (e.g.
    run_trial_daemon.py into trial.status). run_trial() never writes terminal states;
    the caller gets those from the returned TrialResult.
    """
    def _status(state: str, **extra) -> None:
        if on_status is not None:
            on_status(state, extra)

    trial_dir.mkdir(parents=True, exist_ok=True)
    _status("writing_paramfiles")
    try:
        write_paramfiles(params, trial_dir, seed=seed, box_config=box_config)
    except Exception as e:
        return TrialResult(success=False, params=params, error=f"paramfile write failed: {e}")

    # submit_job() already retries sbatch internally, so by the time an exception
    # surfaces here every retry is exhausted. No outer re-run of write_paramfiles():
    # make_class_power() refuses to overwrite an existing class_pk.dat.
    _status("submitting")
    try:
        job_id = submit_job(trial_dir, job_name, box_config=box_config)
    except subprocess.CalledProcessError as e:
        return TrialResult(
            success=False, params=params,
            error=f"sbatch failed after {SUBMIT_MAX_RETRIES} attempts: {e.stderr}",
        )

    _status("running", job_id=job_id)
    job_result = wait_for_job(job_id, trial_dir, stall_timeout_s=box_config.stall_timeout_s)
    cpu_hours = job_result["cpu_hours"]
    wall_min = job_result["wall_time_min"]

    if job_result["state"] != "COMPLETED":
        return TrialResult(
            success=False, params=params, cpu_hours=cpu_hours, wall_time_min=wall_min,
            job_id=job_id, error=f"job ended in state {job_result['state']}",
        )

    try:
        k_sim, pk_sim, nmodes_sim = parse_powerspectrum(trial_dir, scalefactor=box_config.final_scalefactor)
        chi2, reduced_chi2, table = compute_chi2(
            k_sim, pk_sim, nmodes_sim, target_k, target_pk, target_nmodes,
            min_modes=box_config.min_modes, k_max=box_config.k_max_h_per_mpc,
            k_min=box_config.k_min_h_per_mpc, sigma_mode=sigma_mode,
        )
    except Exception as e:
        return TrialResult(
            success=False, params=params, cpu_hours=cpu_hours, wall_time_min=wall_min,
            job_id=job_id, error=f"output parsing/chi2 failed: {e}",
        )

    return TrialResult(
        success=True, params=params, chi2=chi2, reduced_chi2=reduced_chi2,
        n_bins=len(table), table=table, cpu_hours=cpu_hours, wall_time_min=wall_min,
        job_id=job_id,
    )
