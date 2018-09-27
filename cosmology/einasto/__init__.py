#!/usr/bin/env python3
import numpy as np
from scipy.special import gammainc


def Y(u, a):
    return gammainc(3.0 / a, (2.0 / a) * np.power(u, a))


def rho_enc(x, c, a):
    """<rho(x)>/rho_crit"""
    return np.divide(200.0 * Y(c * x, a), np.power(x, 3.0) * Y(c, a))


def m(x, c, a):
    """M(<x)/M_200"""
    return np.divide(Y(c * x, a), Y(c, a))


def m_diff(x, c, a):
    """M(x_{i-1} < x < x_{i})/M_200"""
    y = m(x, c, a)
    y[1:] = np.diff(m(x, c, a))
    return y
