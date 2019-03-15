#!/usr/bin/env python3
import numpy as np

from . import einasto
from . import nfw


# Planck15 values
h_0 = 0.697
rho_0 = 147.7543  # rho_crit(z = 0) in M_solar/kpc^3
OmegaM_0 = 0.281
OmegaL_0 = 1.0 - OmegaM_0
delta_sc_0 = 1.686


def rho_c(z=0.0):
    return rho_0 * (OmegaM_0 * np.power(1.0 + z, 3.0) + OmegaL_0)


def OmegaM(z=0.0):
    return 1.0 - OmegaL(z)


def OmegaL(z=0.0):
    return OmegaL_0 / (OmegaL_0 + OmegaM_0 * np.power(1.0 + z, 3.0))


def Phi(z=0.0):
    return (
        np.power(OmegaM(z), 0.571_428_571_428_571_4)
        - OmegaL(z)
        + (1.0 + 0.5 * OmegaM(z))
        * (1.0 + 0.014_285_714_285_714_285 * OmegaL(z))
    )


def D(z=0.0):
    return (OmegaM(z) * Phi(0.0)) / (OmegaM_0 * Phi(z) * (1.0 + z))


def delta_sc(z=0.0):
    return delta_sc_0 / D(z)


def sigma(m, z=0.0):
    ksi = 1e10 / m
    return (D(z) * 22.26 * np.power(ksi, 0.292)) / (
        1.0 + 1.53 * np.power(ksi, 0.275) + 3.36 * np.power(ksi, 0.198)
    )
