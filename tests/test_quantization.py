import numpy as np
import pytest
from rmq.quantization import optimal_quantizer, lloyd_quantizer, distortion, gradient


@pytest.mark.parametrize("n", [1, 2, 3, 4, 5, 10, 20])
def test_newton_matches_lloyd(n):
    """
    Two independent algorithms (Newton's method and Lloyd's algorithm)
    should converge to the same distortion value, confirming optimal_quantizer
    finds the true optimum rather than a Newton-specific artifact.
    """
    x_newton = optimal_quantizer(n)
    x_lloyd = lloyd_quantizer(n)
    assert distortion(x_newton) == pytest.approx(distortion(x_lloyd), abs=1e-6)


def test_gradient_matches_finite_difference():
    """Analytic gradient should match a central finite-difference approximation."""
    x = np.array([-1.0, 0.0, 1.0])
    eps = 1e-6
    analytic = gradient(x)
    numeric = np.zeros_like(x)
    for i in range(len(x)):
        x_plus = x.copy(); x_plus[i] += eps
        x_minus = x.copy(); x_minus[i] -= eps
        numeric[i] = (distortion(x_plus) - distortion(x_minus)) / (2 * eps)
    assert np.allclose(analytic, numeric, atol=1e-5)


def test_distortion_decreases_with_n():
    """More quantization points should never increase distortion."""
    distortions = [distortion(optimal_quantizer(n)) for n in [1, 2, 3, 5, 10, 20]]
    assert all(d1 > d2 for d1, d2 in zip(distortions, distortions[1:]))