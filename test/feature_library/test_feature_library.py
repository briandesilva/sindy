"""
Unit tests for feature libraries.
"""
import pytest
from scipy.sparse import coo_matrix
from scipy.sparse import csc_matrix
from scipy.sparse import csr_matrix
from sklearn.exceptions import NotFittedError
from sklearn.utils.validation import check_is_fitted

from pysindy.feature_library import ConcatLibrary
from pysindy.feature_library import CustomLibrary
from pysindy.feature_library import FourierLibrary
from pysindy.feature_library import IdentityLibrary
from pysindy.feature_library import PolynomialLibrary
from pysindy.feature_library import SINDyPILibrary
from pysindy.feature_library.base import BaseFeatureLibrary


def test_form_custom_library():
    library_functions = [lambda x: x, lambda x: x ** 2, lambda x: 0 * x]
    function_names = [
        lambda s: str(s),
        lambda s: "{}^2".format(s),
        lambda s: "0",
    ]

    # Test with user-supplied function names
    CustomLibrary(library_functions=library_functions, function_names=function_names)

    # Test without user-supplied function names
    CustomLibrary(library_functions=library_functions, function_names=None)

    # Test with linear control library
    CustomLibrary(
        library_functions=library_functions,
        function_names=None,
        linear_control=True,
        n_control_features=2,
    )

    # Test with user-supplied function names
    SINDyPILibrary(library_functions=library_functions, function_names=function_names)

    # Test without user-supplied function names
    SINDyPILibrary(library_functions=library_functions, function_names=None)


def test_bad_parameters():
    with pytest.raises(ValueError):
        PolynomialLibrary(degree=-1)
    with pytest.raises(ValueError):
        PolynomialLibrary(degree=1.5)
    with pytest.raises(ValueError):
        PolynomialLibrary(include_interaction=False, interaction_only=True)
    with pytest.raises(ValueError):
        FourierLibrary(n_frequencies=-1)
    with pytest.raises(ValueError):
        FourierLibrary(n_frequencies=-1)
    with pytest.raises(ValueError):
        FourierLibrary(n_frequencies=2.2)
    with pytest.raises(ValueError):
        FourierLibrary(include_sin=False, include_cos=False)
    with pytest.raises(ValueError):
        library_functions = [lambda x: x, lambda x: x ** 2, lambda x: 0 * x]
        function_names = [lambda s: str(s), lambda s: "{}^2".format(s)]
        CustomLibrary(
            library_functions=library_functions, function_names=function_names
        )
    with pytest.raises(ValueError):
        library_functions = [lambda x: x]
        function_names = [lambda s: s]
        CustomLibrary(
            library_functions=library_functions,
            function_names=function_names,
            linear_control=True,
        )
    with pytest.raises(ValueError):
        library_functions = [lambda x: x, lambda x: x ** 2, lambda x: 0 * x]
        function_names = [lambda s: str(s), lambda s: "{}^2".format(s)]
        SINDyPILibrary(
            library_functions=library_functions, function_names=function_names
        )
    with pytest.raises(ValueError):
        library_functions = [lambda x: x, lambda x: x ** 2, lambda x: 0 * x]
        function_names = [lambda s: str(s), lambda s: "{}^2".format(s)]
        SINDyPILibrary(
            x_dot_library_functions=library_functions, function_names=function_names
        )
    with pytest.raises(ValueError):
        SINDyPILibrary()
    with pytest.raises(ValueError):
        library_functions = [lambda x: x, lambda x: x ** 2, lambda x: 0 * x]
        SINDyPILibrary(x_dot_library_functions=library_functions)
    with pytest.raises(ValueError):
        library_functions = [lambda x: x, lambda x: x ** 2]
        function_names = [lambda s: s, lambda s: s + s]
        SINDyPILibrary(
            library_functions=library_functions,
            x_dot_library_functions=library_functions,
            function_names=function_names,
        )


@pytest.mark.parametrize(
    "library",
    [
        IdentityLibrary(),
        PolynomialLibrary(),
        FourierLibrary(),
        IdentityLibrary() + PolynomialLibrary(),
        pytest.lazy_fixture("data_custom_library"),
    ],
)
def test_fit_transform(data_lorenz, library):
    x, t = data_lorenz
    library.fit_transform(x)
    check_is_fitted(library)


@pytest.mark.parametrize(
    "library",
    [
        IdentityLibrary(),
        PolynomialLibrary(),
        FourierLibrary(),
        IdentityLibrary() + PolynomialLibrary(),
        pytest.lazy_fixture("data_custom_library"),
    ],
)
def test_change_in_data_shape(data_lorenz, library):
    x, t = data_lorenz
    library.fit(x)
    with pytest.raises(ValueError):
        library.transform(x[:, 1:])


@pytest.mark.parametrize(
    "library, shape",
    [
        (IdentityLibrary(), 3),
        (PolynomialLibrary(), 10),
        (IdentityLibrary() + PolynomialLibrary(), 13),
        (FourierLibrary(), 6),
        (pytest.lazy_fixture("data_custom_library"), 12),
    ],
)
def test_output_shape(data_lorenz, library, shape):
    x, t = data_lorenz
    y = library.fit_transform(x)
    expected_shape = (x.shape[0], shape)
    assert y.shape == expected_shape
    assert library.size > 0


@pytest.mark.parametrize(
    "library",
    [
        IdentityLibrary(),
        PolynomialLibrary(),
        FourierLibrary(),
        PolynomialLibrary() + FourierLibrary(),
        pytest.lazy_fixture("data_custom_library"),
    ],
)
def test_get_feature_names(data_lorenz, library):
    with pytest.raises(NotFittedError):
        library.get_feature_names()

    x, t = data_lorenz
    library.fit_transform(x)
    feature_names = library.get_feature_names()
    assert isinstance(feature_names, list)
    assert isinstance(feature_names[0], str)

    input_features = ["a"] * x.shape[1]
    library.get_feature_names(input_features=input_features)
    assert isinstance(feature_names, list)
    assert isinstance(feature_names[0], str)


@pytest.mark.parametrize("sparse_format", [csc_matrix, csr_matrix, coo_matrix])
def test_polynomial_sparse_inputs(data_lorenz, sparse_format):
    x, t = data_lorenz
    library = PolynomialLibrary()
    library.fit_transform(sparse_format(x))
    check_is_fitted(library)


# Catch-all for various combinations of options and
# inputs for polynomial features
@pytest.mark.parametrize(
    "kwargs, sparse_format",
    [
        ({"degree": 4}, csr_matrix),
        ({"include_bias": True}, csr_matrix),
        ({"include_interaction": False}, lambda x: x),
        ({"include_interaction": False, "include_bias": True}, lambda x: x),
    ],
)
def test_polynomial_options(data_lorenz, kwargs, sparse_format):
    x, t = data_lorenz
    library = PolynomialLibrary(**kwargs)
    library.fit_transform(sparse_format(x))
    check_is_fitted(library)


# Catch-all for various combinations of options and
# inputs for Fourier features
def test_fourier_options(data_lorenz):
    x, t = data_lorenz

    library = FourierLibrary(include_cos=False)
    library.fit_transform(x)
    check_is_fitted(library)


def test_not_implemented(data_lorenz):
    x, t = data_lorenz
    library = BaseFeatureLibrary()

    with pytest.raises(NotImplementedError):
        library.fit(x)

    with pytest.raises(NotImplementedError):
        library.transform(x)

    with pytest.raises(NotImplementedError):
        library.get_feature_names(x)


def test_concat():
    ident_lib = IdentityLibrary()
    poly_lib = PolynomialLibrary()
    concat_lib = ident_lib + poly_lib
    assert isinstance(concat_lib, ConcatLibrary)


@pytest.mark.parametrize(
    "library",
    [
        IdentityLibrary(),
        PolynomialLibrary(),
        FourierLibrary(),
        PolynomialLibrary() + FourierLibrary(),
        pytest.lazy_fixture("data_custom_library"),
    ],
)
def test_not_fitted(data_lorenz, library):
    x, t = data_lorenz

    with pytest.raises(NotFittedError):
        library.transform(x)
