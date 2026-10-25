"""Free, no-HPC proxy for the calibration target: CAMB's nonlinear (halofit) matter
power spectrum. Depends only on the 5 cosmological parameters (Omega0, OmegaBaryon,
Sigma8, HubbleParam, PrimordialIndex) -- it has no AGN/wind knobs, so it can only
constrain the cosmological parameters, never the baryonic ones.
"""
from __future__ import annotations

import numpy as np
import camb

# CAMB has no direct "set sigma8" input: run once at this reference amplitude to
# measure the resulting sigma8, then rescale the amplitude to hit the requested one.
REFERENCE_AS = 2.1e-9


def nonlinear_matter_power(
    k_h_per_mpc: list[float],
    omega0: float,
    omega_baryon: float,
    sigma8: float,
    hubble_param: float,
    primordial_index: float,
    z: float = 0.0,
) -> list[float]:
    omch2 = (omega0 - omega_baryon) * hubble_param ** 2
    ombh2 = omega_baryon * hubble_param ** 2

    pars = camb.CAMBparams()
    pars.set_cosmology(H0=hubble_param * 100, ombh2=ombh2, omch2=omch2)
    pars.InitPower.set_params(ns=primordial_index, As=REFERENCE_AS)
    pars.set_matter_power(redshifts=[z], kmax=max(k_h_per_mpc) * 1.5)
    sigma8_ref = camb.get_results(pars).get_sigma8()[-1]

    as_scaled = REFERENCE_AS * (sigma8 / sigma8_ref) ** 2
    pars.InitPower.set_params(ns=primordial_index, As=as_scaled)
    pars.set_matter_power(redshifts=[z], kmax=max(k_h_per_mpc) * 1.5, nonlinear=True)
    pars.NonLinear = camb.model.NonLinear_both
    results = camb.get_results(pars)

    kh, _, pk = results.get_nonlinear_matter_power_spectrum(hubble_units=True, k_hunit=True)
    return list(np.interp(k_h_per_mpc, kh, pk[0]))
