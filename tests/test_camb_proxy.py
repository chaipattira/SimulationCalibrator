"""Tests for src/camb_proxy.py -- the free, no-HPC CAMB (halofit) nonlinear matter
power spectrum proxy. Runs against the real camb library (no mocking): these numbers
are the actual contract the caller will see.
"""
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import camb_proxy as cp  # noqa: E402

FIDUCIAL = dict(omega0=0.2814, omega_baryon=0.0464, hubble_param=0.7, primordial_index=0.971)


def test_returns_one_value_per_requested_k():
    k = [0.126, 0.3, 0.6, 1.1]
    pk = cp.nonlinear_matter_power(k_h_per_mpc=k, sigma8=0.8, z=0.0, **FIDUCIAL)
    assert len(pk) == len(k)


def test_values_are_positive_finite():
    k = [0.126, 0.3, 0.6, 1.1]
    pk = cp.nonlinear_matter_power(k_h_per_mpc=k, sigma8=0.8, z=0.0, **FIDUCIAL)
    assert all(math.isfinite(v) and v > 0 for v in pk)


def test_low_k_amplitude_scales_as_sigma8_squared():
    # At k=0.01 h/Mpc the spectrum is still deep in the linear regime, where halofit's
    # nonlinear correction is negligible -- P(k) there should scale as sigma8^2 to
    # tight precision, a real physical property (not a mock), so this is a genuine
    # correctness check on the sigma8-rescaling logic.
    k = [0.01]
    pk_lo = cp.nonlinear_matter_power(k_h_per_mpc=k, sigma8=0.8, z=0.0, **FIDUCIAL)[0]
    pk_hi = cp.nonlinear_matter_power(k_h_per_mpc=k, sigma8=0.9, z=0.0, **FIDUCIAL)[0]
    expected_ratio = (0.9 / 0.8) ** 2
    assert math.isclose(pk_hi / pk_lo, expected_ratio, rel_tol=0.01)


def test_higher_sigma8_gives_higher_power_at_k_1p1():
    # Sanity check at the top of a representative k-range (1.1 h/Mpc), not just the
    # linear regime -- confirms the ordering survives into the mildly nonlinear regime.
    k = [1.1]
    pk_lo = cp.nonlinear_matter_power(k_h_per_mpc=k, sigma8=0.8, z=0.0, **FIDUCIAL)[0]
    pk_hi = cp.nonlinear_matter_power(k_h_per_mpc=k, sigma8=0.9, z=0.0, **FIDUCIAL)[0]
    assert pk_hi > pk_lo
