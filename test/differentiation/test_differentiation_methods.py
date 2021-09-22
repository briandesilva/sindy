"""
Unit tests for differentiation methods.
"""
import numpy as np
import pytest
from derivative import dxdt

from pysindy.differentiation import FiniteDifference
from pysindy.differentiation import SINDyDerivative
from pysindy.differentiation import SmoothedFiniteDifference
from pysindy.differentiation.base import BaseDifferentiation


# Simplest example: just use an assert statement
def test_forward_difference_length():
    x = 2 * np.linspace(1, 100, 100)
    forward_difference = FiniteDifference(order=1)
    assert len(forward_difference(x)) == len(x)

    forward_difference_nans = FiniteDifference(order=1, drop_endpoints=True)
    assert len(forward_difference_nans(x)) == len(x)


def test_forward_difference_variable_timestep_length():
    t = np.linspace(1, 10, 100) ** 2
    x = 2 * t
    forward_difference = FiniteDifference(order=1)
    assert len(forward_difference(x, t) == len(x))


def test_centered_difference_length():
    x = 2 * np.linspace(1, 100, 100)
    centered_difference = FiniteDifference(order=2)
    assert len(centered_difference(x)) == len(x)

    centered_difference_nans = FiniteDifference(order=2, drop_endpoints=True)
    assert len(centered_difference_nans(x)) == len(x)


def test_centered_difference_variable_timestep_length():
    t = np.linspace(1, 10, 100) ** 2
    x = 2 * t
    centered_difference = FiniteDifference(order=2)
    assert len(centered_difference(x, t) == len(x))


# Fixtures: data sets to be re-used in multiple tests
# data_derivative_1d and data_derivative_2d are defined
# in ../conftest.py


def test_forward_difference_1d(data_derivative_1d):
    x, x_dot = data_derivative_1d
    forward_difference = FiniteDifference(order=1)
    np.testing.assert_allclose(forward_difference(x), x_dot)


def test_forward_difference_2d(data_derivative_2d):
    x, x_dot = data_derivative_2d
    forward_difference = FiniteDifference(order=1)
    np.testing.assert_allclose(forward_difference(x), x_dot)


def test_centered_difference_1d(data_derivative_1d):
    x, x_dot = data_derivative_1d
    centered_difference = FiniteDifference(order=2)
    np.testing.assert_allclose(centered_difference(x), x_dot)


def test_centered_difference_2d(data_derivative_2d):
    x, x_dot = data_derivative_2d
    centered_difference = FiniteDifference(order=2)
    np.testing.assert_allclose(centered_difference(x), x_dot)


def test_centered_difference_xy_yx(data_2dspatial):
    u = data_2dspatial
    x_grid = np.linspace(1, 100, 100)
    y_grid = np.linspace(1, 50, 50)
    u_x = np.zeros(u.shape)
    u_y = np.zeros(u.shape)
    u_xy = np.zeros(u.shape)
    u_yx = np.zeros(u.shape)
    for i in range(100):
        u_y[i, :, :] = FiniteDifference(order=2, d=1)._centered_difference(
            u[i, :, :], y_grid
        )
    for i in range(50):
        u_x[:, i, :] = FiniteDifference(order=2, d=1)._centered_difference(
            u[:, i, :], x_grid
        )
    for i in range(100):
        u_xy[i, :, :] = FiniteDifference(order=2, d=1)._centered_difference(
            u_x[i, :, :], y_grid
        )
    for i in range(50):
        u_yx[:, i, :] = FiniteDifference(order=2, d=1)._centered_difference(
            u_y[:, i, :], x_grid
        )
    np.testing.assert_allclose(u_xy, u_yx)


# Alternative implementation of the four tests above using parametrization
@pytest.mark.parametrize(
    "data, order",
    [
        (pytest.lazy_fixture("data_derivative_1d"), 1),
        (pytest.lazy_fixture("data_derivative_2d"), 1),
        (pytest.lazy_fixture("data_derivative_1d"), 2),
        (pytest.lazy_fixture("data_derivative_2d"), 2),
    ],
)
def test_finite_difference(data, order):
    x, x_dot = data
    method = FiniteDifference(order=order)
    np.testing.assert_allclose(method(x), x_dot)


# pytest can also check that methods throw errors when appropriate
def test_forward_difference_dim():
    x = np.ones((5, 5, 5))
    forward_difference = FiniteDifference(order=1)
    with pytest.raises(ValueError):
        forward_difference(x)


def test_centered_difference_dim():
    x = np.ones((5, 5, 5))
    centered_difference = FiniteDifference(order=2)
    with pytest.raises(ValueError):
        centered_difference(x)


def test_order_error():
    with pytest.raises(NotImplementedError):
        FiniteDifference(order=3)
    with pytest.raises(ValueError):
        FiniteDifference(order=-1)
    with pytest.raises(ValueError):
        FiniteDifference(d=-1)
    with pytest.raises(ValueError):
        FiniteDifference(d=2, order=1)


def test_base_class(data_derivative_1d):
    x, x_dot = data_derivative_1d
    with pytest.raises(NotImplementedError):
        BaseDifferentiation()._differentiate(x)


# Test smoothed finite difference method
@pytest.mark.parametrize(
    "data",
    [
        pytest.lazy_fixture("data_derivative_1d"),
        pytest.lazy_fixture("data_derivative_2d"),
    ],
)
def test_smoothed_finite_difference(data):
    x, x_dot = data
    smoothed_centered_difference = SmoothedFiniteDifference()
    np.testing.assert_allclose(smoothed_centered_difference(x), x_dot)


@pytest.mark.parametrize(
    "data, derivative_kws",
    [
        (pytest.lazy_fixture("data_derivative_1d"), dict(kind="spectral")),
        (pytest.lazy_fixture("data_derivative_2d"), dict(kind="spectral")),
        (pytest.lazy_fixture("data_derivative_1d"), dict(kind="spline", s=1e-2)),
        (pytest.lazy_fixture("data_derivative_2d"), dict(kind="spline", s=1e-2)),
        (
            pytest.lazy_fixture("data_derivative_1d"),
            dict(kind="trend_filtered", order=0, alpha=1e-2),
        ),
        (
            pytest.lazy_fixture("data_derivative_2d"),
            dict(kind="trend_filtered", order=0, alpha=1e-2),
        ),
        (
            pytest.lazy_fixture("data_derivative_1d"),
            dict(kind="finite_difference", k=1),
        ),
        (
            pytest.lazy_fixture("data_derivative_2d"),
            dict(kind="finite_difference", k=1),
        ),
        (
            pytest.lazy_fixture("data_derivative_1d"),
            dict(kind="savitzky_golay", order=3, left=1, right=1),
        ),
        (
            pytest.lazy_fixture("data_derivative_2d"),
            dict(kind="savitzky_golay", order=3, left=1, right=1),
        ),
    ],
)
def test_wrapper_equivalence_with_dxdt(data, derivative_kws):
    x, _ = data
    t = np.arange(x.shape[0])

    if np.ndim(x) == 1:
        np.testing.assert_allclose(
            dxdt(x.reshape(-1, 1), t, axis=0, **derivative_kws),
            SINDyDerivative(**derivative_kws)(x, t),
        )
    else:
        np.testing.assert_allclose(
            dxdt(x, t, axis=0, **derivative_kws),
            SINDyDerivative(**derivative_kws)(x, t),
        )


@pytest.mark.parametrize(
    "data, derivative_kws",
    [
        (pytest.lazy_fixture("data_derivative_1d"), dict(kind="spectral")),
        (pytest.lazy_fixture("data_derivative_2d"), dict(kind="spectral")),
        (pytest.lazy_fixture("data_derivative_1d"), dict(kind="spline", s=1e-2)),
        (pytest.lazy_fixture("data_derivative_2d"), dict(kind="spline", s=1e-2)),
        (
            pytest.lazy_fixture("data_derivative_1d"),
            dict(kind="trend_filtered", order=0, alpha=1e-2),
        ),
        (
            pytest.lazy_fixture("data_derivative_2d"),
            dict(kind="trend_filtered", order=0, alpha=1e-2),
        ),
        (
            pytest.lazy_fixture("data_derivative_1d"),
            dict(kind="finite_difference", k=1),
        ),
        (
            pytest.lazy_fixture("data_derivative_2d"),
            dict(kind="finite_difference", k=1),
        ),
        (
            pytest.lazy_fixture("data_derivative_1d"),
            dict(kind="savitzky_golay", order=3, left=1, right=1),
        ),
        (
            pytest.lazy_fixture("data_derivative_2d"),
            dict(kind="savitzky_golay", order=3, left=1, right=1),
        ),
    ],
)
def test_derivative_output_shape(data, derivative_kws):
    x, x_dot = data
    t = np.arange(x.shape[0])

    method = SINDyDerivative(**derivative_kws)

    assert x_dot.shape == method(x).shape
    assert x_dot.shape == method(x, t).shape


def test_bad_t_values(data_derivative_1d):
    x, x_dot = data_derivative_1d

    method = SINDyDerivative(kind="finite_difference", k=1)

    with pytest.raises(ValueError):
        method(x, t=-1)

    with pytest.raises(ValueError):
        method._differentiate(x, t=-1)
