from itertools import combinations
from itertools import combinations_with_replacement as combinations_w_r

from numpy import empty
from numpy import hstack
from numpy import nan_to_num
from sklearn.utils import check_array
from sklearn.utils.validation import check_is_fitted

from .base import BaseFeatureLibrary
from pysindy.differentiation import FiniteDifference


class SINDyPILibrary(BaseFeatureLibrary):
    """Generate a library with custom functions. The Library takes custom
    libraries for X and Xdot respectively, and then tensor-products them
    together. For a 3D system, a library of constant and linear terms in x_dot,
    i.e. [1, x_dot0, ..., x_dot3], is good
    enough for most problems and implicit terms. The function names list
    should include both X and Xdot functions, without the mixed terms.

    Parameters
    ----------
    library_functions : list of mathematical functions
        Functions to include in the library. Each function will be
        applied to each input variable x.

    x_dot_library_functions : list of mathematical functions
        Functions to include in the library. Each function will be
        applied to each input variable x_dot.

    t : np.ndarray of time slices
        Time base to compute Xdot from X for the implicit terms

    differentiation_method : differentiation object, optional
        Method for differentiating the data. This must be a class extending
        :class:`pysindy.differentiation_methods.base.BaseDifferentiation` class.
        The default option is centered difference.

    function_names : list of functions, optional (default None)
        List of functions used to generate feature names for each library
        function. Each name function must take a string input (representing
        a variable name), and output a string depiction of the respective
        mathematical function applied to that variable. For example, if the
        first library function is sine, the name function might return
        :math:`\\sin(x)` given :math:`x` as input. The function_names
        list must be the same length as library_functions. If no list of
        function names is provided, defaults to using
        :math:`[ f_0(x),f_1(x), f_2(x), \\ldots ]`. For SINDy-PI,
        function_names should include the names of the functions in both the
        x and x_dot libraries (library_functions and x_dot_library_functions),
        but not the mixed terms, which are computed in the code.

    interaction_only : boolean, optional (default True)
        Whether to omit self-interaction terms.
        If True, function evaulations of the form :math:`f(x,x)` and :math:`f(x,y,x)`
        will be omitted, but those of the form :math:`f(x,y)` and :math:`f(x,y,z)`
        will be included.
        If False, all combinations will be included.

    Attributes
    ----------
    functions : list of functions
        Mathematical library functions to be applied to each input feature.

    function_names : list of functions
        Functions for generating string representations of each library
        function.

    n_input_features_ : int
        The total number of input features.

    n_output_features_ : int
        The total number of output features. The number of output features
        is the product of the number of library functions and the number of
        input features.

    Examples
    --------
    >>> import numpy as np
    >>> from pysindy.feature_library import SINDyPILibrary
    >>> t = np.linspace(0, 1, 5)
    >>> x = np.ones((5, 2))
    >>> functions = [lambda x: 1, lambda x : np.exp(x),
                     lambda x,y : np.sin(x+y)]
    >>> x_dot_functions = [lambda x: 1, lambda x : x]
    >>> function_names = [lambda x: '',
                          lambda x : 'exp(' + x + ')',
                          lambda x, y : 'sin(' + x + y + ')',
                          lambda x: '',
                  lambda x : x]
    >>> lib = ps.SINDyPILibrary(library_functions=functions,
                                x_dot_library_functions=x_dot_functions,
                                function_names=function_names, t=t
                                ).fit(x)
    >>> lib.transform(x)
            [[ 1.00000000e+00  2.71828183e+00  2.71828183e+00  9.09297427e-01
               2.22044605e-16  6.03579815e-16  6.03579815e-16  2.01904588e-16
               2.22044605e-16  6.03579815e-16  6.03579815e-16  2.01904588e-16]
             [ 1.00000000e+00  2.71828183e+00  2.71828183e+00  9.09297427e-01
               0.00000000e+00  0.00000000e+00  0.00000000e+00  0.00000000e+00
               0.00000000e+00  0.00000000e+00  0.00000000e+00  0.00000000e+00]
             [ 1.00000000e+00  2.71828183e+00  2.71828183e+00  9.09297427e-01
               0.00000000e+00  0.00000000e+00  0.00000000e+00  0.00000000e+00
               0.00000000e+00  0.00000000e+00  0.00000000e+00  0.00000000e+00]
             [ 1.00000000e+00  2.71828183e+00  2.71828183e+00  9.09297427e-01
               0.00000000e+00  0.00000000e+00  0.00000000e+00  0.00000000e+00
               0.00000000e+00  0.00000000e+00  0.00000000e+00  0.00000000e+00]
             [ 1.00000000e+00  2.71828183e+00  2.71828183e+00  9.09297427e-01
              -2.22044605e-16 -6.03579815e-16 -6.03579815e-16 -2.01904588e-16
              -2.22044605e-16 -6.03579815e-16 -6.03579815e-16 -2.01904588e-16]]
    >>> lib.get_feature_names()
        ['', 'exp(x0)', 'exp(x1)', 'sin(x0x1)', 'x_dot0', 'exp(x0)x_dot0',
         'exp(x1)x_dot0', 'sin(x0x1)x_dot0', 'x_dot1', 'exp(x0)x_dot1',
         'exp(x1)x_dot1', 'sin(x0x1)x_dot1']
    """

    def __init__(
        self,
        library_functions=None,
        t=None,
        x_dot_library_functions=None,
        function_names=None,
        interaction_only=True,
        differentiation_method=None,
    ):
        super(SINDyPILibrary, self).__init__()
        self.x_functions = library_functions
        self.x_dot_functions = x_dot_library_functions
        self.function_names = function_names
        if library_functions is None and x_dot_library_functions is None:
            raise ValueError(
                "At least one function library, either for x or x_dot, " "is required."
            )
        if x_dot_library_functions is not None and t is None:
            raise ValueError(
                "If using a library that contains x_dot terms,"
                " you must specify a timebase t"
            )
        if function_names is not None:
            x_library_len = 0 if library_functions is None else len(library_functions)
            x_dot_library_len = (
                0 if x_dot_library_functions is None else len(x_dot_library_functions)
            )
            if x_library_len + x_dot_library_len != len(function_names):
                raise ValueError(
                    "(x_library_functions + x_dot_library_functions) and "
                    " function_names must have the same"
                    " number of elements"
                )
        if differentiation_method is None and x_dot_library_functions is not None:
            differentiation_method = FiniteDifference(drop_endpoints=False)
        self.differentiation_method = differentiation_method
        self.interaction_only = interaction_only
        self.t = t

    @staticmethod
    def _combinations(n_features, n_args, interaction_only):
        """Get the combinations of features to be passed to a library function."""
        comb = combinations if interaction_only else combinations_w_r
        return comb(range(n_features), n_args)

    def get_feature_names(self, input_features=None):
        """Return feature names for output features.

        Parameters
        ----------
        input_features : list of string, length n_features, optional
            String names for input features if available. By default,
            "x0", "x1", ... "xn_features" is used.

        Returns
        -------
        output_feature_names : list of string, length n_output_features
        """
        check_is_fitted(self)
        x_dot_features = ["x_dot%d" % i for i in range(self.n_input_features_)]
        if input_features is None:
            input_features = ["x%d" % i for i in range(self.n_input_features_)]
        feature_names = []
        if self.x_dot_functions is not None and self.x_functions is not None:
            for k, fdot in enumerate(self.x_dot_functions):
                if (
                    k == 0
                    and self.function_names[-len(self.x_dot_functions)]("0") == ""
                ):
                    n_x_dot_input_features_ = 1
                else:
                    n_x_dot_input_features_ = self.n_input_features_
                for fdot_combs in self._combinations(
                    n_x_dot_input_features_,
                    fdot.__code__.co_argcount,
                    self.interaction_only,
                ):
                    for i, f in enumerate(self.x_functions):
                        if i == 0 and self.function_names[0]("0") == "":
                            nx_input_features_ = 1
                        else:
                            nx_input_features_ = self.n_input_features_
                        for f_combs in self._combinations(
                            nx_input_features_,
                            f.__code__.co_argcount,
                            self.interaction_only,
                        ):
                            feature_names.append(
                                self.function_names[i](
                                    *[input_features[comb] for comb in f_combs]
                                )
                                + self.function_names[len(self.x_functions) + k](
                                    *[x_dot_features[comb] for comb in fdot_combs]
                                )
                            )
        else:
            if self.x_dot_functions is None:
                funcs = self.x_functions
            else:
                funcs = self.x_dot_functions
            for i, f in enumerate(funcs):
                for c in self._combinations(
                    self.n_input_features_,
                    f.__code__.co_argcount,
                    self.interaction_only,
                ):
                    feature_names.append(
                        self.function_names[i](*[input_features[j] for j in c])
                    )
        return feature_names

    def fit(self, x, y=None):
        """Compute number of output features.

        Parameters
        ----------
        x : array-like, shape (n_samples, n_features)
            Measurement data.

        Returns
        -------
        self : instance
        """
        n_samples, n_features = check_array(x).shape
        self.n_input_features_ = n_features
        n_x_output_features = 0
        n_x_dot_output_features = 0
        if self.x_dot_functions is not None and self.x_functions is not None:
            for i, f in enumerate(self.x_functions):
                if i == 0 and (self.function_names[0]("0") == ""):
                    nx_input_features_ = 1
                else:
                    nx_input_features_ = self.n_input_features_
                n_args = f.__code__.co_argcount
                n_x_output_features += len(
                    list(
                        self._combinations(
                            nx_input_features_, n_args, self.interaction_only
                        )
                    )
                )
            for i, fdot in enumerate(self.x_dot_functions):
                if i == 0 and (
                    self.function_names[-len(self.x_dot_functions)]("0") == ""
                ):
                    n_x_dot_input_features_ = 1
                else:
                    n_x_dot_input_features_ = self.n_input_features_
                n_args = fdot.__code__.co_argcount
                n_x_dot_output_features += len(
                    list(
                        self._combinations(
                            n_x_dot_input_features_, n_args, self.interaction_only
                        )
                    )
                )
            self.n_output_features_ = n_x_output_features * n_x_dot_output_features
        else:
            if self.x_dot_functions is None:
                funcs = self.x_functions
            else:
                funcs = self.x_dot_functions
            for f in funcs:
                n_args = f.__code__.co_argcount
                n_x_output_features += len(
                    list(self._combinations(n_features, n_args, self.interaction_only))
                )
            self.n_output_features_ = n_x_output_features
        if self.function_names is None:
            self.function_names = list(
                map(
                    lambda i: (lambda *x: "f" + str(i) + "(" + ",".join(x) + ")"),
                    range(len(self.x_functions)),
                )
            )
            self.function_names = hstack(
                (
                    self.function_names,
                    list(
                        map(
                            lambda i: (
                                lambda *x_dot: "fdot"
                                + str(i)
                                + "("
                                + ",".join(x_dot)
                                + ")"
                            ),
                            range(len(self.x_dot_functions)),
                        )
                    ),
                )
            )
        return self

    def transform(self, x):
        """Transform data to custom features

        Parameters
        ----------
        x : array-like, shape (n_samples, n_features)
            The data to transform, row by row.

        Returns
        -------
        xp : np.ndarray, shape (n_samples, n_output_features)
            The matrix of features, where n_output_features is the number of features
            generated from applying the custom functions to the inputs.
        """
        check_is_fitted(self)

        x = check_array(x)
        if self.x_dot_functions is not None:
            x_dot = nan_to_num(self.differentiation_method(x, self.t))

        n_samples, n_features = x.shape

        if n_features != self.n_input_features_:
            raise ValueError("x shape does not match training shape")

        xp = empty((n_samples, self.n_output_features_), dtype=x.dtype)
        library_idx = 0
        if self.x_dot_functions is not None and self.x_functions is not None:
            for k, fdot in enumerate(self.x_dot_functions):
                if k == 0 and (
                    self.function_names[-len(self.x_dot_functions)]("0") == ""
                ):
                    n_x_dot_input_features_ = 1
                else:
                    n_x_dot_input_features_ = self.n_input_features_
                for fdot_combs in self._combinations(
                    n_x_dot_input_features_,
                    fdot.__code__.co_argcount,
                    self.interaction_only,
                ):

                    for i, f in enumerate(self.x_functions):
                        if i == 0 and (self.function_names[0]("0") == ""):
                            nx_input_features_ = 1
                        else:
                            nx_input_features_ = self.n_input_features_
                        for f_combs in self._combinations(
                            nx_input_features_,
                            f.__code__.co_argcount,
                            self.interaction_only,
                        ):
                            xp[:, library_idx] = f(
                                *[x[:, comb] for comb in f_combs]
                            ) * fdot(*[x_dot[:, comb] for comb in fdot_combs])
                            library_idx += 1
        else:
            if self.x_dot_functions is None:
                funcs = self.x_functions
            else:
                funcs = self.x_dot_functions
            for f in funcs:
                for c in self._combinations(
                    self.n_input_features_,
                    f.__code__.co_argcount,
                    self.interaction_only,
                ):
                    xp[:, library_idx] = f(*[x[:, j] for j in c])
                    library_idx += 1
        return xp
