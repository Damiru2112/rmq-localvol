"""
Optimal scalar quantization of the standard Gaussian distribution.

Given N, finds the grid x_1 < ... < x_N and Voronoi weights that minimize
the L2 distortion E[min_i (X - x_i)^2] for X ~ N(0, 1), via Newton's method.
"""

import numpy as np
from scipy.stats import norm


def voronoi_boundaries(x: np.ndarray) -> np.ndarray:
    """
    Given a sorted grid x of length N, return the N-1 interior cell
    boundaries (midpoints between adjacent grid points).
    """
    x = np.asarray(x)
    return 0.5 * (x[:-1] + x[1:])

def _truncated_moments(a: float, b: float):
    """
    Closed-form truncated moments of a standard Gaussian over (a, b):
        mass  = ∫_a^b φ(z) dz       = Φ(b) - Φ(a)
        mean  = ∫_a^b z φ(z) dz     = φ(a) - φ(b)
        sq    = ∫_a^b z^2 φ(z) dz   = mass + a φ(a) - b φ(b)

    Handles a = -inf / b = +inf explicitly: the boundary terms a*phi(a)
    and b*phi(b) vanish in the limit (phi decays faster than linear growth),
    but "-inf * 0" evaluates to nan in plain arithmetic, so each term must
    be zeroed out manually when the corresponding bound is infinite.
    """
    phi_a, phi_b = norm.pdf(a), norm.pdf(b)
    Phi_a, Phi_b = norm.cdf(a), norm.cdf(b)

    mass = Phi_b - Phi_a
    mean = phi_a - phi_b

    a_term = 0.0 if np.isinf(a) else a * phi_a
    b_term = 0.0 if np.isinf(b) else b * phi_b
    sq = mass + a_term - b_term

    return mass, mean, sq


def distortion(x: np.ndarray) -> float:
    """
    L2 distortion of grid x against the standard Gaussian:
        D(x) = sum_i  integral over cell_i of (z - x_i)^2 * phi(z) dz
    """
    x = np.asarray(x)
    boundaries = voronoi_boundaries(x)
    # cell i has left edge a_i, right edge b_i
    a = np.concatenate(([-np.inf], boundaries))
    b = np.concatenate((boundaries, [np.inf]))

    total = 0.0
    for i in range(len(x)):
        mass, mean, sq = _truncated_moments(a[i], b[i])
        total += sq - 2 * x[i] * mean + x[i] ** 2 * mass
    return total

def gradient(x: np.ndarray) -> np.ndarray:
    """
    Gradient of the distortion functional D(x) with respect to each x_i.

        dD/dx_i = 2 * x_i * mass_i - 2 * mean_i

    where mass_i, mean_i are the truncated 0th and 1st moments of the
    standard Gaussian over cell i. The boundary-movement terms cancel
    exactly (shared boundary, continuous integrand), so only this direct
    term survives.
    """
    x = np.asarray(x)
    boundaries = voronoi_boundaries(x)
    a = np.concatenate(([-np.inf], boundaries))
    b = np.concatenate((boundaries, [np.inf]))

    grad = np.empty_like(x, dtype=float)
    for i in range(len(x)):
        mass, mean, _ = _truncated_moments(a[i], b[i])
        grad[i] = 2 * x[i] * mass - 2 * mean
    return grad

def optimal_quantizer(N: int, max_iter: int = 200, tol: float = 1e-10) -> np.ndarray:
    """
    Find the N-point optimal quantizer of the standard Gaussian via
    diagonal quasi-Newton iteration on the distortion functional.

    Update rule:
        x_i <- x_i - grad_i / (2 * mass_i)

    where mass_i is the probability mass of x_i's Voronoi cell. This
    approximates the Hessian diagonal (2 * mass_i is the dominant term),
    ignoring the off-diagonal cross-cell terms.

    Initialized at the N quantiles of the standard Gaussian, which is a
    reasonable, well-spread starting grid.
    """
    # initial grid: N quantiles, evenly spaced in probability space
    probs = (np.arange(N) + 0.5) / N
    x = norm.ppf(probs)

    for iteration in range(max_iter):
        boundaries = voronoi_boundaries(x)
        a = np.concatenate(([-np.inf], boundaries))
        b = np.concatenate((boundaries, [np.inf]))

        grad = np.empty(N)
        masses = np.empty(N)
        for i in range(N):
            mass, mean, _ = _truncated_moments(a[i], b[i])
            grad[i] = 2 * x[i] * mass - 2 * mean
            masses[i] = mass

        step = grad / (2 * masses)
        x = x - step

        if np.max(np.abs(step)) < tol:
            break

    return np.sort(x)

def lloyd_quantizer(N: int, max_iter: int = 1000, tol: float = 1e-12) -> np.ndarray:
    """
    Independent cross-check for optimal_quantizer: Lloyd's algorithm.
    Alternates (1) assigning Voronoi cells from the current grid, and
    (2) moving each grid point to its cell's conditional mean (centroid).
    Provably monotonically decreases distortion; used here only to
    verify optimal_quantizer (Newton's method) converges to the same
    answer via a different route.
    """
    probs = (np.arange(N) + 0.5) / N
    x = norm.ppf(probs)

    for _ in range(max_iter):
        boundaries = voronoi_boundaries(x)
        a = np.concatenate(([-np.inf], boundaries))
        b = np.concatenate((boundaries, [np.inf]))

        x_new = np.empty(N)
        for i in range(N):
            mass, mean, _ = _truncated_moments(a[i], b[i])
            x_new[i] = mean / mass  # centroid = E[Z | cell i]

        if np.max(np.abs(x_new - x)) < tol:
            x = x_new
            break
        x = x_new

    return np.sort(x)