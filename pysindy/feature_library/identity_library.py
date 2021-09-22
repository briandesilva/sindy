from numpy import delete
from sklearn.utils import check_array
from sklearn.utils.validation import check_is_fitted

from .base import BaseFeatureLibrary


class IdentityLibrary(BaseFeatureLibrary):
    """
    Generate an identity library which maps all input features to
    themselves.

    Attributes
    ----------
    n_input_features_ : int
        The total number of input features.

    n_output_features_ : int
        The total number of output features. The number of output features
        is equal to the number of input features.

    library_ensemble : boolean, optional (default False)
        Whether or not to do library bagging.

    Examples
    --------
    >>> import numpy as np
    >>> from pysindy.feature_library import IdentityLibrary
    >>> x = np.array([[0,-1],[0.5,-1.5],[1.,-2.]])
    >>> lib = IdentityLibrary().fit(x)
    >>> lib.transform(x)
    array([[ 0. , -1. ],
           [ 0.5, -1.5],
           [ 1. , -2. ]])
    >>> lib.get_feature_names()
    ['x0', 'x1']
    """

    def __init__(
        self,
        library_ensemble=False,
        ensemble_indices=0,
    ):
        super(IdentityLibrary, self).__init__()
        self.library_ensemble = library_ensemble
        self.ensemble_indices = ensemble_indices

    def get_feature_names(self, input_features=None):
        """
        Return feature names for output features

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
        if input_features:
            if len(input_features) == self.n_input_features_:
                return input_features
            else:
                raise ValueError("input features list is not the right length")
        return ["x%d" % i for i in range(self.n_input_features_)]

    def fit(self, x, y=None):
        """
        Compute number of output features.

        Parameters
        ----------
        x : array-like, shape (n_samples, n_features)
            The data.

        Returns
        -------
        self : instance
        """
        n_samples, n_features = check_array(x).shape
        self.n_input_features_ = n_features
        self.n_output_features_ = n_features
        return self

    def transform(self, x):
        """Perform identity transformation (return a copy of the input).

        Parameters
        ----------
        x : array-like, shape (n_samples, n_features)
            The data to transform, row by row.

        Returns
        -------
        x : np.ndarray, shape (n_samples, n_features)
            The matrix of features, which is just a copy of the input data.
        """
        check_is_fitted(self)

        x = check_array(x)

        n_samples, n_features = x.shape

        if n_features != self.n_input_features_:
            raise ValueError("x shape does not match training shape")

        # If library bagging, return x missing a single column
        if self.library_ensemble:
            if n_features == 1:
                raise ValueError(
                    "Can't use library ensemble methods if your"
                    " library is just one term!"
                )
            inds = range(self.n_output_features_)
            inds = delete(inds, self.ensemble_indices)
            return (x.copy())[:, inds]
        else:
            return x.copy()
