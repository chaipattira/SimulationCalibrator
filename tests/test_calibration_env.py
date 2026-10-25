"""Tests for src/calibration_env.py's BoxConfig support, and that the STAR_SMALL_BOX
default reproduces the module's standalone constants/behavior -- the call pattern that
passes no box_config explicitly.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import calibration_env as env  # noqa: E402


def test_star_small_box_matches_existing_module_constants():
    box = env.STAR_SMALL_BOX
    assert box.template_dir == env.STAR_SMALL_DIR
    assert box.box_size_mpc_h == env.BOX_SIZE_MPC_H
    assert box.ic_ngrid == env.IC_NGRID
    assert box.final_scalefactor == env.FINAL_SCALEFACTOR
    assert box.min_modes == env.MIN_MODES
    assert box.k_max_h_per_mpc == env.K_MAX_H_PER_MPC
    assert box.k_min_h_per_mpc == 0.0
    assert box.partition == env.SLURM_PARTITION
    assert box.ntasks == env.NTASKS
    assert box.cpus_per_task == env.CPUS_PER_TASK
    assert box.mem == env.MEM
    assert box.job_time == env.JOB_TIME


def test_render_sbatch_script_uses_default_box_config_values():
    script = env._render_sbatch_script("calib-test", env.STAR_SMALL_BOX)
    assert f"--partition={env.SLURM_PARTITION}" in script
    assert f"--ntasks={env.NTASKS}" in script
    assert f"--mem={env.MEM}" in script
    assert f"--time={env.JOB_TIME}" in script


def test_render_sbatch_script_uses_custom_box_config_values():
    custom = env.BoxConfig(
        name="test-hr",
        template_dir=env.STAR_SMALL_DIR,  # irrelevant to sbatch rendering
        box_size_mpc_h=50.0,
        ic_ngrid=128,
        final_scalefactor="1.0000",
        min_modes=20,
        k_max_h_per_mpc=1.1,
        k_min_h_per_mpc=0.126,
        partition="cpu",
        ntasks=16,
        cpus_per_task=2,
        mem="64G",
        job_time="24:00:00",
    )
    script = env._render_sbatch_script("calib-hr-test", custom)
    assert "--partition=cpu" in script
    assert "--ntasks=16" in script
    assert "--mem=64G" in script
    assert "--time=24:00:00" in script


def test_write_paramfiles_reads_from_custom_box_config_template_dir(tmp_path):
    # Minimal fixture template dir with a sentinel comment that would never appear
    # if write_paramfiles silently fell back to the default STAR_SMALL_DIR.
    template_dir = tmp_path / "fixture-template"
    template_dir.mkdir()
    (template_dir / "paramfile.genic").write_text(
        "# SENTINEL fixture template\n"
        "Ngrid = 128\n"
        "BoxSize = 50000\n"
        "Omega0 = 0.3\n"
        "OmegaLambda = 0.7\n"
        "OmegaBaryon = 0.05\n"
        "HubbleParam = 0.7\n"
        "Redshift = 9\n"
        "Sigma8 = 0.8\n"
        "Seed = 1\n"
        "FileWithInputSpectrum = class_pk.dat\n"
        "DifferentTransferFunctions = 0\n"
    )
    (template_dir / "paramfile.gadget").write_text(
        "# SENTINEL fixture template\n"
        "Omega0 = 0.3\n"
        "OmegaLambda = 0.7\n"
        "OmegaBaryon = 0.05\n"
    )
    custom = env.BoxConfig(
        name="fixture", template_dir=template_dir, box_size_mpc_h=50.0, ic_ngrid=128,
        final_scalefactor="1.0000", min_modes=1, k_max_h_per_mpc=10.0,
    )
    rundir = tmp_path / "run"

    env.write_paramfiles(
        {"Omega0": 0.31, "OmegaBaryon": 0.049, "Sigma8": 0.81,
         "WindSigma0": 300.0, "WindSpeedFactor": 3.0, "BlackHoleFeedbackFactor": 0.05},
        rundir, box_config=custom,
    )

    genic_text = (rundir / "paramfile.genic").read_text()
    gadget_text = (rundir / "paramfile.gadget").read_text()
    assert "SENTINEL fixture template" in genic_text
    assert "SENTINEL fixture template" in gadget_text
    assert (rundir / "class_pk.dat").exists()


def test_compute_chi2_trims_bins_below_k_min():
    k = [0.05, 0.2, 0.5]
    pk = [100.0, 50.0, 10.0]
    nmodes = [50, 50, 50]
    _chi2, _reduced, table = env.compute_chi2(
        k, pk, nmodes, k, pk, nmodes, min_modes=1, k_max=10.0, k_min=0.1,
    )
    assert [row["k"] for row in table] == [0.2, 0.5]


def test_write_paramfiles_default_box_config_still_uses_real_star_small_template(tmp_path):
    # Regression guard: no box_config passed (the six-key call pattern) must still
    # read the real examples/star-small templates and succeed.
    env.write_paramfiles(
        {"Omega0": 0.288, "OmegaBaryon": 0.0472, "Sigma8": 0.80,
         "WindSigma0": 353.0, "WindSpeedFactor": 3.7, "BlackHoleFeedbackFactor": 0.05},
        tmp_path / "run",
    )
    gadget_text = (tmp_path / "run" / "paramfile.gadget").read_text()
    assert "WindSigma0 = 353.0" in gadget_text
    assert (tmp_path / "run" / "class_pk.dat").exists()


def test_compute_chi2_k_min_defaults_to_zero_backward_compatible():
    k = [0.05, 0.2, 0.5]
    pk = [100.0, 50.0, 10.0]
    nmodes = [50, 50, 50]
    _chi2, _reduced, table = env.compute_chi2(
        k, pk, nmodes, k, pk, nmodes, min_modes=1, k_max=10.0,
    )
    assert len(table) == 3
