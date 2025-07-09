import numpy as np

def fit_sin_alone(x, A, omega, phi, offset):
    return A * np.sin((omega * x) + phi) + offset

def fit_sin_and_lin(x, A, omega, phi, offset, m):
    return A * np.sin((omega * x) + phi) + (m * x) + offset

def fit_sin_and_quad(x, A, omega, phi, offset, a, b):
    return A * np.sin((omega * x) + phi) + (b * x) + (a * x**2) + offset

def fit_sin_and_cubic(x, A, omega, phi, offset, a, b, c):
    return A * np.sin((omega * x) + phi) + (c * x**3) + (b * x**2) + (a * x) + offset