"""
Unit tests for SINDy class.

Note: all tests should be encapsulated in functions whose
names start with "test_"

To run all tests for this package, navigate to the top-level
directory and execute the following command:
pytest

To run tests for just one file, run
pytest file_to_test.py

"""
import numpy as np
import pytest
from sklearn.exceptions import ConvergenceWarning
from sklearn.exceptions import NotFittedError
from sklearn.linear_model import ElasticNet
from sklearn.linear_model import Lasso
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import TimeSeriesSplit
from sklearn.utils.validation import check_is_fitted

from pysindy import SINDy
from pysindy.differentiation import FiniteDifference
from pysindy.differentiation import SINDyDerivative
from pysindy.differentiation import SmoothedFiniteDifference
from pysindy.feature_library import FourierLibrary
from pysindy.feature_library import PolynomialLibrary
from pysindy.optimizers import ConstrainedSR3
from pysindy.optimizers import SR3
from pysindy.optimizers import STLSQ


def test_get_feature_names_len(data_lorenz):
    x, t = data_lorenz
    model = SINDy()

    with pytest.raises(NotFittedError):
        model.get_feature_names()

    model.fit(x, t)

    # Assumes default library is polynomial features of degree 2
    assert len(model.get_feature_names()) == 10


def test_not_fitted(data_1d):
    x, t = data_1d
    model = SINDy()

    with pytest.raises(NotFittedError):
        model.predict(x)
    with pytest.raises(NotFittedError):
        model.get_feature_names()
    with pytest.raises(NotFittedError):
        model.coefficients()
    with pytest.raises(NotFittedError):
        model.equations()
    with pytest.raises(NotFittedError):
        model.simulate(x[0], t)


def test_improper_shape_input(data_1d):
    x, t = data_1d

    # Ensure model successfully handles different data shapes
    model = SINDy()
    model.fit(x.flatten(), t)
    check_is_fitted(model)

    model = SINDy()
    model.fit(x.flatten(), t, x_dot=x.flatten())
    check_is_fitted(model)

    model = SINDy()
    model.fit(x, t, x_dot=x.flatten())
    check_is_fitted(model)


def test_nan_derivatives(data_lorenz):
    x, t = data_lorenz

    model = SINDy(differentiation_method=FiniteDifference(drop_endpoints=True))
    model.fit(x, t)
    check_is_fitted(model)


@pytest.mark.parametrize(
    "data",
    [
        pytest.lazy_fixture("data_1d"),
        pytest.lazy_fixture("data_lorenz"),
        pytest.lazy_fixture("data_1d_bad_shape"),
    ],
)
def test_mixed_inputs(data):
    x, t = data

    # Scalar t
    model = SINDy()
    model.fit(x, t=2)
    check_is_fitted(model)

    # x_dot is passed in
    model = SINDy()
    model.fit(x, x_dot=x)
    check_is_fitted(model)

    model = SINDy()
    model.fit(x, t, x_dot=x)
    check_is_fitted(model)


@pytest.mark.parametrize(
    "data", [pytest.lazy_fixture("data_1d"), pytest.lazy_fixture("data_lorenz")]
)
def test_bad_t(data):
    x, t = data
    model = SINDy()

    # Wrong type
    with pytest.raises(ValueError):
        model.fit(x, t="1")

    # Invalid value of t
    with pytest.raises(ValueError):
        model.fit(x, t=-1)

    # t is a list
    with pytest.raises(ValueError):
        model.fit(x, list(t))

    # Wrong number of time points
    with pytest.raises(ValueError):
        model.fit(x, t[:-1])

    # Two points in t out of order
    t[2], t[4] = t[4], t[2]
    with pytest.raises(ValueError):
        model.fit(x, t)
    t[2], t[4] = t[4], t[2]

    # Two matching times in t
    t[3] = t[5]
    with pytest.raises(ValueError):
        model.fit(x, t)


@pytest.mark.parametrize(
    "data", [pytest.lazy_fixture("data_1d"), pytest.lazy_fixture("data_lorenz")]
)
def test_t_default(data):
    x, t = data
    dt = t[1] - t[0]

    with pytest.raises(ValueError):
        model = SINDy(t_default=0)
    with pytest.raises(ValueError):
        model = SINDy(t_default="1")

    model = SINDy()
    model.fit(x, dt)

    model_t_default = SINDy(t_default=dt)
    model_t_default.fit(x)

    np.testing.assert_allclose(model.coefficients(), model_t_default.coefficients())
    np.testing.assert_almost_equal(model.score(x, t=dt), model_t_default.score(x))
    np.testing.assert_almost_equal(
        model.differentiate(x, t=dt), model_t_default.differentiate(x)
    )


@pytest.mark.parametrize(
    "data", [pytest.lazy_fixture("data_1d"), pytest.lazy_fixture("data_lorenz")]
)
@pytest.mark.parametrize(
    "optimizer",
    [
        STLSQ(),
        SR3(),
        ConstrainedSR3(),
        Lasso(fit_intercept=False),
        ElasticNet(fit_intercept=False),
    ],
)
def test_predict(data, optimizer):
    x, t = data
    model = SINDy(optimizer=optimizer)
    model.fit(x, t)
    x_dot = model.predict(x)

    assert x.shape == x_dot.shape


@pytest.mark.parametrize(
    "data",
    [
        pytest.lazy_fixture("data_1d"),
        pytest.lazy_fixture("data_lorenz"),
        pytest.lazy_fixture("data_1d_bad_shape"),
    ],
)
def test_simulate(data):
    x, t = data
    model = SINDy()
    model.fit(x, t)
    x1 = model.simulate(np.ravel(x[0]), t)

    assert len(x1) == len(t)


@pytest.mark.parametrize(
    "library",
    [
        PolynomialLibrary(degree=3),
        FourierLibrary(n_frequencies=3),
        pytest.lazy_fixture("data_custom_library"),
        PolynomialLibrary() + FourierLibrary(),
    ],
)
def test_libraries(data_lorenz, library):
    x, t = data_lorenz
    model = SINDy(feature_library=library)
    model.fit(x, t)

    s = model.score(x, t)
    assert s <= 1


def test_integration_smoothed_finite_difference(data_lorenz):
    x, t = data_lorenz
    model = SINDy(differentiation_method=SmoothedFiniteDifference())

    model.fit(x, t=t)

    check_is_fitted(model)


@pytest.mark.parametrize(
    "derivative_kws",
    [
        dict(kind="finite_difference", k=1),
        dict(kind="spectral"),
        dict(kind="spline", s=1e-2),
        dict(kind="trend_filtered", order=0, alpha=1e-2),
        dict(kind="savitzky_golay", order=3, left=1, right=1),
    ],
)
def test_integration_derivative_methods(data_lorenz, derivative_kws):
    x, t = data_lorenz
    model = SINDy(differentiation_method=SINDyDerivative(**derivative_kws))
    model.fit(x, t=t)

    check_is_fitted(model)


@pytest.mark.parametrize(
    "data",
    [
        pytest.lazy_fixture("data_1d"),
        pytest.lazy_fixture("data_lorenz"),
        pytest.lazy_fixture("data_1d_bad_shape"),
    ],
)
def test_score(data):
    x, t = data
    model = SINDy()
    model.fit(x, t)

    assert model.score(x) <= 1

    assert model.score(x, t) <= 1

    assert model.score(x, x_dot=x) <= 1

    assert model.score(x, t, x_dot=x) <= 1


def test_fit_multiple_trajectores(data_multiple_trajctories):
    x, t = data_multiple_trajctories
    model = SINDy()

    # Should fail if multiple_trajectories flag is not set
    with pytest.raises(ValueError):
        model.fit(x, t=t)

    model.fit(x, multiple_trajectories=True)
    check_is_fitted(model)

    model.fit(x, t=t, multiple_trajectories=True)
    assert model.score(x, t=t, multiple_trajectories=True) > 0.8

    model = SINDy()
    model.fit(x, x_dot=x, multiple_trajectories=True)
    check_is_fitted(model)

    model = SINDy()
    model.fit(x, t=t, x_dot=x, multiple_trajectories=True)
    check_is_fitted(model)

    # Test validate_input
    t[0] = None
    with pytest.raises(ValueError):
        model.fit(x, t=t, multiple_trajectories=True)


def test_predict_multiple_trajectories(data_multiple_trajctories):
    x, t = data_multiple_trajctories
    model = SINDy()
    model.fit(x, t=t, multiple_trajectories=True)

    # Should fail if multiple_trajectories flag is not set
    with pytest.raises(ValueError):
        model.predict(x)

    p = model.predict(x, multiple_trajectories=True)
    assert len(p) == len(x)


def test_score_multiple_trajectories(data_multiple_trajctories):
    x, t = data_multiple_trajctories
    model = SINDy()
    model.fit(x, t=t, multiple_trajectories=True)

    # Should fail if multiple_trajectories flag is not set
    with pytest.raises(ValueError):
        model.score(x)

    s = model.score(x, multiple_trajectories=True)
    assert s <= 1

    s = model.score(x, t=t, multiple_trajectories=True)
    assert s <= 1

    s = model.score(x, x_dot=x, multiple_trajectories=True)
    assert s <= 1

    s = model.score(x, t=t, x_dot=x, multiple_trajectories=True)
    assert s <= 1


def test_fit_discrete_time(data_discrete_time):
    x = data_discrete_time

    model = SINDy(discrete_time=True)
    model.fit(x)
    check_is_fitted(model)

    model = SINDy(discrete_time=True)
    model.fit(x[:-1], x_dot=x[1:])
    check_is_fitted(model)


def test_simulate_discrete_time(data_discrete_time):
    x = data_discrete_time
    model = SINDy(discrete_time=True)
    model.fit(x)
    n_steps = x.shape[0]
    x1 = model.simulate(x[0], n_steps)

    assert len(x1) == n_steps

    # TODO: implement test using the stop_condition option


def test_predict_discrete_time(data_discrete_time):
    x = data_discrete_time
    model = SINDy(discrete_time=True)
    model.fit(x)
    assert len(model.predict(x)) == len(x)


def test_score_discrete_time(data_discrete_time):
    x = data_discrete_time
    model = SINDy(discrete_time=True)
    model.fit(x)
    assert model.score(x) > 0.75
    assert model.score(x, x_dot=x) < 1


def test_fit_discrete_time_multiple_trajectories(
    data_discrete_time_multiple_trajectories,
):
    x = data_discrete_time_multiple_trajectories

    # Should fail if multiple_trajectories flag is not set
    model = SINDy(discrete_time=True)
    with pytest.raises(ValueError):
        model.fit(x)

    model.fit(x, multiple_trajectories=True)
    check_is_fitted(model)

    model = SINDy(discrete_time=True)
    model.fit(x, x_dot=x, multiple_trajectories=True)
    check_is_fitted(model)


def test_predict_discrete_time_multiple_trajectories(
    data_discrete_time_multiple_trajectories,
):
    x = data_discrete_time_multiple_trajectories
    model = SINDy(discrete_time=True)
    model.fit(x, multiple_trajectories=True)

    # Should fail if multiple_trajectories flag is not set
    with pytest.raises(ValueError):
        model.predict(x)

    y = model.predict(x, multiple_trajectories=True)
    assert len(y) == len(x)


def test_score_discrete_time_multiple_trajectories(
    data_discrete_time_multiple_trajectories,
):
    x = data_discrete_time_multiple_trajectories
    model = SINDy(discrete_time=True)
    model.fit(x, multiple_trajectories=True)

    # Should fail if multiple_trajectories flag is not set
    with pytest.raises(ValueError):
        model.score(x)

    s = model.score(x, multiple_trajectories=True)
    assert s > 0.75

    # x is not its own derivative, so we expect bad performance here
    s = model.score(x, x_dot=x, multiple_trajectories=True)
    assert s < 1


@pytest.mark.parametrize(
    "data",
    [
        pytest.lazy_fixture("data_1d"),
        pytest.lazy_fixture("data_lorenz"),
        pytest.lazy_fixture("data_1d_bad_shape"),
    ],
)
def test_equations(data, capsys):
    x, t = data
    model = SINDy()
    model.fit(x, t)

    out, _ = capsys.readouterr()
    assert len(out) == 0

    model.print(precision=2)

    out, _ = capsys.readouterr()
    assert len(out) > 0


def test_print_discrete_time(data_discrete_time, capsys):
    x = data_discrete_time
    model = SINDy(discrete_time=True)
    model.fit(x)
    model.print()

    out, _ = capsys.readouterr()
    assert len(out) > 0


def test_print_discrete_time_multiple_trajectories(
    data_discrete_time_multiple_trajectories, capsys
):
    x = data_discrete_time_multiple_trajectories
    model = SINDy(discrete_time=True)
    model.fit(x, multiple_trajectories=True)

    model.print()

    out, _ = capsys.readouterr()
    assert len(out) > 1


def test_differentiate(data_lorenz, data_multiple_trajctories):
    x, t = data_lorenz

    model = SINDy()
    model.differentiate(x, t)

    x, t = data_multiple_trajctories
    model.differentiate(x, t, multiple_trajectories=True)

    model = SINDy(discrete_time=True)
    with pytest.raises(RuntimeError):
        model.differentiate(x)


def test_coefficients(data_lorenz):
    x, t = data_lorenz
    model = SINDy()
    model.fit(x, t)
    c = model.coefficients()
    assert np.count_nonzero(c) < 10


def test_complexity(data_lorenz):
    x, t = data_lorenz
    model = SINDy()
    model.fit(x, t)
    assert model.complexity < 10


def test_multiple_trajectories_errors(data_multiple_trajctories, data_discrete_time):
    x, t = data_multiple_trajctories

    model = SINDy()
    with pytest.raises(TypeError):
        model._process_multiple_trajectories(np.array(x, dtype=object), t, x)
    with pytest.raises(TypeError):
        model._process_multiple_trajectories(x, t, np.array(x, dtype=object))

    # Test an option that doesn't get tested elsewhere
    model._process_multiple_trajectories(x, t, x, return_array=False)

    x = data_discrete_time
    model = SINDy(discrete_time=True)
    with pytest.raises(TypeError):
        model._process_multiple_trajectories(x, t, np.array(x, dtype=object))


def test_simulate_errors(data_lorenz):
    x, t = data_lorenz
    model = SINDy()
    model.fit(x, t)

    with pytest.raises(ValueError):
        model.simulate(x[0], t=1)

    model = SINDy(discrete_time=True)
    with pytest.raises(ValueError):
        model.simulate(x[0], t=[1, 2])


@pytest.mark.parametrize(
    "params, warning",
    [({"threshold": 100}, UserWarning), ({"max_iter": 1}, ConvergenceWarning)],
)
def test_fit_warn(data_lorenz, params, warning):
    x, t = data_lorenz
    model = SINDy(optimizer=STLSQ(**params))

    with pytest.warns(warning):
        model.fit(x, t)

    with pytest.warns(None) as warn_record:
        model.fit(x, t, quiet=True)

    assert len(warn_record) == 0


def test_cross_validation(data_lorenz):
    x, t = data_lorenz
    dt = t[1] - t[0]

    model = SINDy(
        t_default=dt, differentiation_method=SINDyDerivative(kind="spline", s=1e-2)
    )

    param_grid = {
        "optimizer__threshold": [0.01, 0.1],
        "differentiation_method__kwargs": [
            {"kind": "spline", "s": 1e-2},
            {"kind": "finite_difference", "k": 1},
        ],
        "feature_library__degree": [1, 2],
    }

    search = RandomizedSearchCV(
        model, param_grid, cv=TimeSeriesSplit(n_splits=3), n_iter=5
    )
    search.fit(x)
    check_is_fitted(search)


def test_linear_constraints(data_lorenz):
    x, t = data_lorenz

    library = PolynomialLibrary().fit(x)

    constraint_rhs = np.ones(2)
    constraint_lhs = np.zeros((2, x.shape[1] * library.n_output_features_))

    target_1, target_2 = 1, 3
    constraint_lhs[0, 3] = target_1
    constraint_lhs[1, library.n_output_features_] = target_2

    optimizer = ConstrainedSR3(
        constraint_lhs=constraint_lhs, constraint_rhs=constraint_rhs
    )
    model = SINDy(feature_library=library, optimizer=optimizer).fit(x, t)

    coeffs = model.coefficients()

    np.testing.assert_allclose(
        np.array([coeffs[0, 3], coeffs[1, 0]]), np.array([1 / target_1, 1 / target_2])
    )
