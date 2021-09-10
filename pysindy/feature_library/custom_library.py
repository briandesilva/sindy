from itertools import combinations
from itertools import combinations_with_replacement as combinations_w_r

from numpy import empty
from numpy import shape
from sklearn.utils import check_array
from sklearn.utils.validation import check_is_fitted

from .base import BaseFeatureLibrary


class CustomLibrary(BaseFeatureLibrary):
    """Generate a library with custom functions.

    Parameters
    ----------
    library_functions : list of mathematical functions
        Functions to include in the library. Default is to use same functions
        for all variables. Can also be used so that each variable has an
        associated library, in this case library_functions is shape
        (n_input_features, num_library_functions)

    function_names : list of functions, optional (default None)
        List of functions used to generate feature names for each library
        function. Each name function must take a string input (representing
        a variable name), and output a string depiction of the respective
        mathematical function applied to that variable. For example, if the
        first library function is sine, the name function might return
        :math:`\\sin(x)` given :math:`x` as input. The function_names list must be the
        same length as library_functions. If no list of function names is
        provided, defaults to using :math:`[ f_0(x),f_1(x), f_2(x), \\ldots ]`.

    interaction_only : boolean, optional (default True)
        Whether to omit self-interaction terms.
        If True, function evaulations of the form :math:`f(x,x)` and :math:`f(x,y,x)`
        will be omitted, but those of the form :math:`f(x,y)` and :math:`f(x,y,z)`
        will be included.
        If False, all combinations will be included.

    linear_control : boolean, optional (default False)
        Special option to allow for a pure linear control term
        in the SINDy library. The default option is to take a tensor product
        of the control term u, with all library terms in library_functions. So
        a quadratic polynomial library would give you
        [1, x, x^2, u, ux, ux^2]. If linear_control is True, it produces only
        the pure linear u term, i.e. [1, x, x^2, u].

    n_control_features : int, optional (default None)
        If linear_control = True, then this specifies the shape
        of the control input

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
    >>> from pysindy.feature_library import CustomLibrary
    >>> x = np.array([[0.,-1],[1.,0.],[2.,-1.]])
    >>> functions = [lambda x : np.exp(x), lambda x,y : np.sin(x+y)]
    >>> lib = CustomLibrary(library_functions=functions).fit(x)
    >>> lib.transform(x)
    array([[ 1.        ,  0.36787944, -0.84147098],
           [ 2.71828183,  1.        ,  0.84147098],
           [ 7.3890561 ,  0.36787944,  0.84147098]])
    >>> lib.get_feature_names()
    ['f0(x0)', 'f0(x1)', 'f1(x0,x1)']
    """

    def __init__(
        self,
        library_functions,
        function_names=None,
        interaction_only=True,
        linear_control=False,
        n_control_features=None,
    ):
        super(CustomLibrary, self).__init__()
        self.functions = library_functions
        self.function_names = function_names
        self.linear_control = linear_control
        self.n_control_features = n_control_features
        if function_names and (
            shape(library_functions)[-1] != shape(function_names)[-1]
        ):
            raise ValueError(
                "library_functions and function_names must have the same"
                " number of elements"
            )
        if linear_control and n_control_features is None:
            raise ValueError(
                "If using linear control option, need to pass "
                "n_control_features argument"
            )
        self.interaction_only = interaction_only

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
        n_features = self.n_input_features_
        if self.linear_control:
            n_features = self.n_input_features_ - self.n_control_features
        if input_features is None:
            input_features = ["x%d" % i for i in range(n_features)]
        feature_names = []
        for i, f in enumerate(self.functions):
            for c in self._combinations(
                n_features, f.__code__.co_argcount, self.interaction_only
            ):
                feature_names.append(
                    self.function_names[i](*[input_features[j] for j in c])
                )
        if self.linear_control:
            for i in range(self.n_control_features):
                feature_names.append("u%d" % i)
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
        if self.linear_control:
            n_x = n_features - self.n_control_features
        else:
            n_x = n_features
        n_output_features = 0
        for f in self.functions:
            n_args = f.__code__.co_argcount
            n_output_features += len(
                list(self._combinations(n_x, n_args, self.interaction_only))
            )
            self.n_output_features_ = n_output_features
            if self.function_names is None:
                self.function_names = list(
                    map(
                        lambda i: (lambda *x: "f" + str(i) + "(" + ",".join(x) + ")"),
                        range(len(self.functions)),
                    )
                )
        if self.linear_control:
            self.n_output_features_ += self.n_control_features
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

        n_samples, n_features = x.shape

        if n_features != self.n_input_features_:
            raise ValueError("x shape does not match training shape")
        if self.linear_control:
            n_x = self.n_input_features_ - self.n_control_features
        else:
            n_x = self.n_input_features_
        xp = empty((n_samples, self.n_output_features_), dtype=x.dtype)
        library_idx = 0
        for f in self.functions:
            for c in self._combinations(
                n_x, f.__code__.co_argcount, self.interaction_only
            ):
                xp[:, library_idx] = f(*[x[:, j] for j in c])
                library_idx += 1
        if self.linear_control:
            for i in range(n_x, self.n_input_features_):
                xp[:, library_idx] = x[:, i]
                library_idx += 1
        return xp
