#!/usr/bin/env python3
import numpy as np
from scipy.optimize import root
from scipy.special import erf

from . import einasto
from . import nfw


# Planck15 values
h_0 = 0.697
rho_0 = 147.7543  # rho_crit(z = 0) in M_solar/kpc^3
OmegaM_0 = 0.281
OmegaL_0 = 1.0 - OmegaM_0
delta_sc_0 = 1.686
c = 299_792_458  # m/sec


def rho_c(z=0.0):
    return rho_0 * (OmegaM_0 * np.power(1.0 + z, 3.0) + OmegaL_0)


def OmegaM(z=0.0):
    return 1.0 - OmegaL(z)


def OmegaL(z=0.0):
    return np.divide(OmegaL_0, OmegaL_0 + OmegaM_0 * np.power(1 + z, 3))


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


def fR(z=0.0, n=1):
    """f_R(z) / f_R0 for Hu-Sawicki f(R) gravity
    """
    return np.power(
        np.divide(
            1 + 4 * np.divide(OmegaL(z), OmegaM(z)),
            np.power(1 + z, 3) + 4 * np.divide(OmegaL(z), OmegaM(z)),
        ),
        n + 1,
    )


def p2(fR, z=0.0):
    """p2 mass rescaling for f(R) gravity (Mitchell+2019)
    """
    return 1.503 * np.log10(np.divide(fR, 1 + z)) + 21.64


def m500_from_m200(c, m200):
    return (
        m200
        * root(
            lambda x: np.log10(cosmology.nfw.rho_enc(x, c=c)) - np.log10(500),
            0.7,
        ).x[0]
    )


def correction_mitchell(c, m, z, fR0):
    """Mitchell+2019 model
    """
    l, k_s, o_s, a, g, o_t, k_t, sqrt2 = (
        0.458,
        -0.324,
        1.49,
        -6.17,
        -0.039,
        0.82,
        0.01,
        np.sqrt(2.0),
    )

    x = m500_from_m200(c, m) - p2(fR0 * fR(z), z)
    xp = (x - k_s) / o_s

    return (
        0.5
        * (
            (
                (l / o_s)
                * 0.398_942_280_401_432_7  # 1/sqrt(2pi)
                * np.exp(-0.5 * xp * xp)
                * (1 + erf((a * xp) / sqrt2))
            )
            + g
        )
        * (1 - np.tanh(o_t * (x + k_t)))
    )
