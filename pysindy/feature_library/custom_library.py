from itertools import combinations
from itertools import combinations_with_replacement as combinations_w_r

from numpy import delete
from numpy import empty
from sklearn.utils import check_array
from sklearn.utils.validation import check_is_fitted

from .base import BaseFeatureLibrary


class CustomLibrary(BaseFeatureLibrary):
    """Generate a library with custom functions.

    Parameters
    ----------
    library_functions : list of mathematical functions
        Functions to include in the library. Each function will be
        applied to each input variable.

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

    library_ensemble : boolean, optional (default False)
        Whether or not to do library bagging.

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
        library_ensemble=False,
        ensemble_indices=0,
    ):
        super(CustomLibrary, self).__init__()
        self.functions = library_functions
        self.function_names = function_names
        if function_names and (len(library_functions) != len(function_names)):
            raise ValueError(
                "library_functions and function_names must have the same"
                " number of elements"
            )
        self.interaction_only = interaction_only
        self.library_ensemble = library_ensemble
        self.ensemble_indices = ensemble_indices

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
        if input_features is None:
            input_features = ["x%d" % i for i in range(self.n_input_features_)]
        feature_names = []
        for i, f in enumerate(self.functions):
            for c in self._combinations(
                self.n_input_features_, f.__code__.co_argcount, self.interaction_only
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
        n_output_features = 0
        for f in self.functions:
            n_args = f.__code__.co_argcount
            n_output_features += len(
                list(self._combinations(n_features, n_args, self.interaction_only))
            )
        self.n_output_features_ = n_output_features
        if self.function_names is None:
            self.function_names = list(
                map(
                    lambda i: (lambda *x: "f" + str(i) + "(" + ",".join(x) + ")"),
                    range(len(self.functions)),
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

        n_samples, n_features = x.shape

        if n_features != self.n_input_features_:
            raise ValueError("x shape does not match training shape")

        xp = empty((n_samples, self.n_output_features_), dtype=x.dtype)
        library_idx = 0
        for f in self.functions:
            for c in self._combinations(
                self.n_input_features_, f.__code__.co_argcount, self.interaction_only
            ):
                xp[:, library_idx] = f(*[x[:, j] for j in c])
                library_idx += 1

        # If library bagging, return xp missing a single column
        if self.library_ensemble:
            if self.n_output_features_ == 1:
                raise ValueError(
                    "Can't use library ensemble methods if your"
                    " library is just one term!"
                )
            inds = range(self.n_output_features_)
            inds = delete(inds, self.ensemble_indices)
            return xp[:, inds]
        else:
            return xp
