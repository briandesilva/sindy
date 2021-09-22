import numpy as np
from sklearn.utils import check_array
from sklearn.utils.validation import check_is_fitted

from .base import BaseFeatureLibrary


class FourierLibrary(BaseFeatureLibrary):
    """
    Generate a library with trigonometric functions.

    Parameters
    ----------
    n_frequencies : int, optional (default 1)
        Number of frequencies to include in the library. The library will
        include functions :math:`\\sin(x), \\sin(2x), \\dots \\sin(n_{frequencies}x)`
        for each input feature :math:`x` (depending on which of sine and/or cosine
        features are included).

    include_sin : boolean, optional (default True)
        If True, include sine terms in the library.

    include_cos : boolean, optional (default True)
        If True, include cosine terms in the library.

    library_ensemble : boolean, optional (default False)
        Whether or not to do library bagging.

    Attributes
    ----------
    n_input_features_ : int
        The total number of input features.

    n_output_features_ : int
        The total number of output features. The number of output features
        is ``2 * n_input_features_ * n_frequencies`` if both sines and cosines
        are included. Otherwise it is ``n_input_features * n_frequencies``.

    Examples
    --------
    >>> import numpy as np
    >>> from pysindy.feature_library import FourierLibrary
    >>> x = np.array([[0.],[1.],[2.]])
    >>> lib = FourierLibrary(n_frequencies=2).fit(x)
    >>> lib.transform(x)
    array([[ 0.        ,  1.        ,  0.        ,  1.        ],
           [ 0.84147098,  0.54030231,  0.90929743, -0.41614684],
           [ 0.90929743, -0.41614684, -0.7568025 , -0.65364362]])
    >>> lib.get_feature_names()
    ['sin(1 x0)', 'cos(1 x0)', 'sin(2 x0)', 'cos(2 x0)']
    """

    def __init__(
        self,
        n_frequencies=1,
        include_sin=True,
        include_cos=True,
        library_ensemble=False,
        ensemble_indices=0,
    ):
        super(FourierLibrary, self).__init__()
        if not (include_sin or include_cos):
            raise ValueError("include_sin and include_cos cannot both be False")
        if n_frequencies < 1 or not isinstance(n_frequencies, int):
            raise ValueError("n_frequencies must be a positive integer")
        self.n_frequencies = n_frequencies
        self.include_sin = include_sin
        self.include_cos = include_cos
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
        if input_features is None:
            input_features = ["x%d" % i for i in range(self.n_input_features_)]
        feature_names = []
        for i in range(self.n_frequencies):
            for feature in input_features:
                if self.include_sin:
                    feature_names.append("sin(" + str(i + 1) + " " + feature + ")")
                if self.include_cos:
                    feature_names.append("cos(" + str(i + 1) + " " + feature + ")")
        return feature_names

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
        if self.include_sin and self.include_cos:
            self.n_output_features_ = n_features * self.n_frequencies * 2
        else:
            self.n_output_features_ = n_features * self.n_frequencies
        return self

    def transform(self, x):
        """Transform data to Fourier features

        Parameters
        ----------
        x : array-like, shape (n_samples, n_features)
            The data to transform, row by row.

        Returns
        -------
        xp : np.ndarray, shape (n_samples, n_output_features)
            The matrix of features, where n_output_features is the number of Fourier
            features generated from the inputs.
        """
        check_is_fitted(self)

        x = check_array(x)

        n_samples, n_features = x.shape

        if n_features != self.n_input_features_:
            raise ValueError("x shape does not match training shape")

        xp = np.empty((n_samples, self.n_output_features_), dtype=x.dtype)
        idx = 0
        for i in range(self.n_frequencies):
            for j in range(self.n_input_features_):
                if self.include_sin:
                    xp[:, idx] = np.sin((i + 1) * x[:, j])
                    idx += 1
                if self.include_cos:
                    xp[:, idx] = np.cos((i + 1) * x[:, j])
                    idx += 1

        # If library bagging, return xp missing a single column
        if self.library_ensemble:
            if self.n_output_features_ == 1:
                raise ValueError(
                    "Can't use library ensemble methods if your"
                    " library is just one term!"
                )
            inds = range(self.n_output_features_)
            inds = np.delete(inds, self.ensemble_indices)
            return xp[:, inds]
        else:
            return xp
