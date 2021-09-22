from itertools import combinations
from itertools import combinations_with_replacement as combinations_w_r

from numpy import asarray
from numpy import delete
from numpy import empty
from numpy import hstack
from numpy import linspace
from numpy import meshgrid
from numpy import ones
from numpy import ravel
from numpy import reshape
from numpy import shape
from numpy import transpose
from numpy import zeros
from numpy.random import uniform
from scipy.integrate import trapezoid
from scipy.interpolate import RectBivariateSpline
from scipy.interpolate import RegularGridInterpolator
from scipy.special import hyp2f1
from scipy.special import poch
from sklearn.utils import check_array
from sklearn.utils.validation import check_is_fitted

from .base import BaseFeatureLibrary
from pysindy.differentiation import FiniteDifference


class PDELibrary(BaseFeatureLibrary):
    """Generate a PDE library with custom functions.

    Parameters
    ----------
    library_functions : list of mathematical functions
        Functions to include in the library. Each function will be
        applied to each input variable (but not their derivatives)

    derivative_order : int, optional (default 0)
        Order of derivative to take on each input variable, max 4

    spatial_grid : np.ndarray, optional (default None)
        The spatial grid for computing derivatives

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

    include_bias : boolean, optional (default True)
        If True (default), then include a bias column, the feature in which
        all polynomial powers are zero (i.e. a column of ones - acts as an
        intercept term in a linear model).

    is_uniform : boolean, optional (default True)
        If True, assume the grid is uniform in all spatial directions, so
        can use uniform grid spacing for the derivative calculations.

    weak_form : boolean, optional (default False)
        If True, uses the weak/integral form of SINDy, requiring some extra
        parameters.

    K : int, optional (default 100)
        Number of domain centers, corresponding to subdomain squares of length
        Hx and height Hy. If weak_form is True but K is not
        specified, defaults to 100.

    Hx : float, optional (default None)
        Half of the length of the square subdomains. If weak_form is True
        but Hx is not specified, defaults to Hx = Lx / 20, where
        Lx is the length of the full domain.

    Hy : float, optional (default None)
        Half of the height of the square subdomains. If weak_form is True
        but Hy is not specified, defaults to Hy = Ly / 20, where
        Ly is the height of the full domain.

    Ht : float, optional (default None)
        Half of the temporal length of the square subdomains.
        If weak_form is True but Ht is not specified, defaults to
        Ht = M / 20, where M is the number of time points.

    p : int, optional (default 4)
        Positive integer to define the polynomial degree of the spatial weights
        used for weak/integral SINDy.

    library_ensemble : boolean, optional (default False)
        Whether or not to use library bagging (regress on subset of the
        candidate terms in the library)

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
    >>> from pysindy.feature_library import PDELibrary
    >>> x = np.array([[0.,-1],[1.,0.],[2.,-1.]])
    >>> functions = [lambda x : np.exp(x), lambda x,y : np.sin(x+y)]
    >>> lib = PDELibrary(library_functions=functions).fit(x)
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
        derivative_order=0,
        spatial_grid=None,
        temporal_grid=None,
        function_names=None,
        interaction_only=True,
        include_bias=True,
        is_uniform=False,
        weak_form=False,
        K=100,
        num_pts_per_domain=100,
        Hx=None,
        Hy=None,
        Ht=None,
        p=4,
        library_ensemble=False,
        ensemble_indices=0,
    ):
        super(PDELibrary, self).__init__()
        self.derivative_order = derivative_order
        self.spatial_grid = spatial_grid
        self.temporal_grid = temporal_grid
        self.functions = library_functions
        self.function_names = function_names
        self.library_ensemble = library_ensemble
        self.ensemble_indices = ensemble_indices
        self.include_bias = include_bias
        self.is_uniform = is_uniform
        self.num_pts_per_domain = num_pts_per_domain
        if function_names and (len(library_functions) != len(function_names)):
            raise ValueError(
                "library_functions and function_names must have the same"
                " number of elements"
            )
        if derivative_order < 0 or derivative_order > 4:
            raise ValueError("The derivative order must be 1, 2, 3, or 4")
        if (spatial_grid is not None and derivative_order == 0) or (
            spatial_grid is None and derivative_order != 0
        ):
            raise ValueError(
                "Spatial grid and the derivative order must be "
                "defined at the same time"
            )
        if spatial_grid is not None and (
            (len(spatial_grid.shape) != 1)
            and (len(spatial_grid.shape) != 3)
            and (len(spatial_grid.shape) != 4)
        ):
            raise ValueError("Spatial grid size is incorrect")
        self.interaction_only = interaction_only
        if spatial_grid is not None:
            self.slen = len((self.spatial_grid).shape)
            if self.slen == 1:
                self.num_derivs = derivative_order
            elif self.slen == 3:
                num_derivs = 2
                for i in range(2, derivative_order + 1):
                    num_derivs += i + 1
                self.num_derivs = num_derivs

        # weak form checks now
        if weak_form and temporal_grid is not None:
            if len(shape(temporal_grid)) != 1:
                raise ValueError("Temporal grid must be 1D.")
            self.M = len(temporal_grid)
            t1 = temporal_grid[0]
            t2 = temporal_grid[-1]
            if Ht is not None:
                if Ht <= 0:
                    raise ValueError("Ht must be a positive float")
                if Ht >= (t2 - t1) / 2.0:
                    raise ValueError("2 * Ht is larger than the time domain")
                self.Ht = Ht
            else:
                Lt = t2 - t1
                self.Ht = Lt / 20.0
        if weak_form and (spatial_grid is None or temporal_grid is None):
            raise ValueError(
                "Weak form requires user to pass both spatial and " "temporal grids."
            )

        self.weak_form = weak_form
        self.num_pts_per_domain = num_pts_per_domain
        self.p = p
        if weak_form and spatial_grid is not None:
            if p < 0:
                raise ValueError("Poly degree of the spatial weights must be > 0")
            if Hx is not None:
                if Hx <= 0:
                    raise ValueError("Hx must be a positive float")
                if self.slen == 1:
                    x1 = spatial_grid[0]
                    x2 = spatial_grid[-1]
                if self.slen == 3:
                    x1 = spatial_grid[0, 0, 0]
                    x2 = spatial_grid[-1, 0, 0]
                if Hx >= (x2 - x1) / 2.0:
                    raise ValueError("2 * Hx is bigger than the full domain length")
                self.Hx = Hx
            else:
                if self.slen == 1:
                    x1 = spatial_grid[0]
                    x2 = spatial_grid[-1]
                if self.slen == 3:
                    x1 = spatial_grid[0, 0, 0]
                    x2 = spatial_grid[-1, 0, 0]
                Lx = x2 - x1
                self.Hx = Lx / 20.0
            if Hy is not None and self.slen == 3:
                if Hy <= 0:
                    raise ValueError("Hy must be a positive float")
                y1 = spatial_grid[0, 0, 1]
                y2 = spatial_grid[0, -1, 1]
                if Hy >= (y2 - y1) / 2.0:
                    raise ValueError("2 * Hy is bigger than the full domain height")
                self.Hy = Hy
            elif Hy is None and self.slen == 3:
                y1 = spatial_grid[0, 0, 0]
                y2 = spatial_grid[0, -1, 1]
                Ly = y2 - y1
                self.Hy = Ly / 20.0
            if K <= 0:
                raise ValueError("The number of subdomains must be > 0")
            self.K = K
            if self.slen == 1:
                x1 = spatial_grid[0]
                x2 = spatial_grid[-1]
                domain_centersx = uniform(x1 + self.Hx, x2 - self.Hx, size=(self.K, 1))
                domain_centerst = uniform(t1 + self.Ht, t2 - self.Ht, size=(self.K, 1))
                domain_centers = hstack((domain_centersx, domain_centerst))
            if self.slen == 3:
                x1 = spatial_grid[0, 0, 0]
                x2 = spatial_grid[-1, 0, 0]
                y1 = spatial_grid[0, 0, 1]
                y2 = spatial_grid[0, -1, 1]
                domain_centersx = uniform(x1 + self.Hx, x2 - self.Hx, size=(self.K, 1))
                domain_centersy = uniform(y1 + self.Hy, y2 - self.Hy, size=(self.K, 1))
                domain_centerst = uniform(t1 + self.Ht, t2 - self.Ht, size=(self.K, 1))
                domain_centers = hstack((domain_centersx, domain_centersy))
                domain_centers = hstack((domain_centers, domain_centerst))
            self.domain_centers = domain_centers

    @staticmethod
    def _combinations(n_features, n_args, interaction_only):
        """Get the combinations of features to be passed to a library function."""
        comb = combinations if interaction_only else combinations_w_r
        return comb(range(n_features), n_args)

    def _poly_deriv_1D(self, x, t, d_x, d_t):
        """Compute analytic derivatives instead of relying on finite diffs"""
        x_term = (
            (2 * x) ** d_x
            * (x ** 2 - 1) ** (self.p - d_x)
            * hyp2f1((1 - d_x) / 2.0, -d_x / 2.0, self.p + 1 - d_x, 1 - 1 / x ** 2)
            * poch(self.p + 1 - d_x, d_x)
            / self.Hx ** d_x
        )
        t_term = (
            (2 * t) ** d_t
            * (t ** 2 - 1) ** (self.p - d_t)
            * hyp2f1((1 - d_t) / 2.0, -d_t / 2.0, self.p + 1 - d_t, 1 - 1 / t ** 2)
            * poch(self.p + 1 - d_t, d_t)
            / self.Ht ** d_t
        )
        return x_term * t_term

    def _poly_deriv_2D(self, x, y, t, d_x, d_y, d_t):
        """Compute analytic derivatives instead of relying on finite diffs"""
        x_term = (
            (2 * x) ** d_x
            * (x ** 2 - 1) ** (self.p - d_x)
            * hyp2f1((1 - d_x) / 2.0, -d_x / 2.0, self.p + 1 - d_x, 1 - 1 / x ** 2)
            * poch(self.p + 1 - d_x, d_x)
            / self.Hx ** d_x
        )
        y_term = (
            (2 * y) ** d_y
            * (y ** 2 - 1) ** (self.p - d_y)
            * hyp2f1((1 - d_y) / 2.0, -d_y / 2.0, self.p + 1 - d_y, 1 - 1 / y ** 2)
            * poch(self.p + 1 - d_y, d_y)
            / self.Hy ** d_y
        )
        t_term = (
            (2 * t) ** d_t
            * (t ** 2 - 1) ** (self.p - d_t)
            * hyp2f1((1 - d_t) / 2.0, -d_t / 2.0, self.p + 1 - d_t, 1 - 1 / t ** 2)
            * poch(self.p + 1 - d_t, d_t)
            / self.Ht ** d_t
        )
        return x_term * y_term * t_term

    def _smooth_ppoly(self, x, t, k, d_x, d_y, d_t):
        if shape(x)[1] == 1:
            x_tilde = zeros(shape(x)[0])
            t_tilde = zeros(shape(t)[0])
            weights = zeros((shape(x)[0], shape(t)[0], 1))
            for i in range(shape(x)[0]):
                x_tilde[i] = (x[i] - self.domain_centers[k, 0]) / self.Hx
                t_tilde[i] = (t[i] - self.domain_centers[k, -1]) / self.Ht
            for i in range(shape(x)[0]):
                weights[i, :, 0] = self._poly_deriv_1D(x_tilde[i], t_tilde, d_x, d_t)
            return weights
        if shape(x)[1] == 2:
            x_tilde = zeros(shape(x)[0])
            y_tilde = zeros(shape(x)[0])
            t_tilde = zeros(shape(t)[0])
            for i in range(shape(x)[0]):
                x_tilde[i] = (x[i, 0] - self.domain_centers[k, 0]) / self.Hx
                y_tilde[i] = (x[i, 1] - self.domain_centers[k, 1]) / self.Hy
                t_tilde[i] = (t[i] - self.domain_centers[k, -1]) / self.Ht
            weights = zeros((shape(x)[0], shape(x)[0], shape(t)[0], 1))
            for i in range(shape(x)[0]):
                for j in range(shape(x)[0]):
                    weights[i, j, :, 0] = self._poly_deriv_2D(
                        x_tilde[i], y_tilde[j], t_tilde, d_x, d_y, d_t
                    )
            return weights

    def _make_2D_derivatives(self, u):
        (num_gridx, num_gridy, num_time, n_features) = u.shape
        u_derivs = zeros((num_gridx, num_gridy, num_time, n_features, self.num_derivs))
        # first derivatives
        for i in range(num_time):
            # u_x
            for kk in range(num_gridy):
                u_derivs[:, kk, i, :, 0] = FiniteDifference(
                    d=1, is_uniform=self.is_uniform
                )._differentiate(u[:, kk, i, :], self.spatial_grid[:, kk, 0])
            # u_y
            for kk in range(num_gridx):
                u_derivs[kk, :, i, :, 1] = FiniteDifference(
                    d=1, is_uniform=self.is_uniform
                )._differentiate(u[kk, :, i, :], self.spatial_grid[kk, :, 1])

        if self.derivative_order >= 2:
            # second derivatives
            for i in range(num_time):
                # u_xx
                for kk in range(num_gridy):
                    u_derivs[:, kk, i, :, 2] = FiniteDifference(
                        d=2, is_uniform=self.is_uniform
                    )._differentiate(u[:, kk, i, :], self.spatial_grid[:, kk, 0])
                # u_xy
                for kk in range(num_gridx):
                    u_derivs[kk, :, i, :, 3] = FiniteDifference(
                        d=1, is_uniform=self.is_uniform
                    )._differentiate(
                        u_derivs[kk, :, i, :, 0], self.spatial_grid[kk, :, 1]
                    )
                # u_yy
                for kk in range(num_gridx):
                    u_derivs[kk, :, i, :, 4] = FiniteDifference(
                        d=2, is_uniform=self.is_uniform
                    )._differentiate(u[kk, :, i, :], self.spatial_grid[kk, :, 1])

        if self.derivative_order >= 3:
            # third derivatives
            for i in range(num_time):
                # u_xxx
                for kk in range(num_gridy):
                    u_derivs[:, kk, i, :, 5] = FiniteDifference(
                        d=3, is_uniform=self.is_uniform
                    )._differentiate(u[:, kk, i, :], self.spatial_grid[:, kk, 0])
                # u_xxy
                for kk in range(num_gridx):
                    u_derivs[kk, :, i, :, 6] = FiniteDifference(
                        d=1, is_uniform=self.is_uniform
                    )._differentiate(
                        u_derivs[kk, :, i, :, 2], self.spatial_grid[kk, :, 1]
                    )
                # u_xyy
                for kk in range(num_gridx):
                    u_derivs[kk, :, i, :, 7] = FiniteDifference(
                        d=1, is_uniform=self.is_uniform
                    )._differentiate(
                        u_derivs[kk, :, i, :, 3], self.spatial_grid[kk, :, 1]
                    )
                # u_yyy
                for kk in range(num_gridx):
                    u_derivs[kk, :, i, :, 8] = FiniteDifference(
                        d=3, is_uniform=self.is_uniform
                    )._differentiate(u[kk, :, i, :], self.spatial_grid[kk, :, 1])

        if self.derivative_order >= 4:
            # fourth derivatives
            for i in range(num_time):
                # u_xxxx
                for kk in range(num_gridy):
                    u_derivs[:, kk, i, :, 9] = FiniteDifference(
                        d=4, is_uniform=self.is_uniform
                    )._differentiate(u[:, kk, i, :], self.spatial_grid[:, kk, 0])
                # u_xxxy
                for kk in range(num_gridx):
                    u_derivs[kk, :, i, :, 10] = FiniteDifference(
                        d=1, is_uniform=self.is_uniform
                    )._differentiate(
                        u_derivs[kk, :, i, :, 5], self.spatial_grid[kk, :, 1]
                    )
                # u_xxyy
                for kk in range(num_gridx):
                    u_derivs[kk, :, i, :, 11] = FiniteDifference(
                        d=1, is_uniform=self.is_uniform
                    )._differentiate(
                        u_derivs[kk, :, i, :, 6], self.spatial_grid[kk, :, 1]
                    )
                # u_xyyy
                for kk in range(num_gridx):
                    u_derivs[kk, :, i, :, 12] = FiniteDifference(
                        d=1, is_uniform=self.is_uniform
                    )._differentiate(
                        u_derivs[kk, :, i, :, 7], self.spatial_grid[kk, :, 1]
                    )
                # u_yyyy
                for kk in range(num_gridx):
                    u_derivs[kk, :, i, :, 13] = FiniteDifference(
                        d=4, is_uniform=self.is_uniform
                    )._differentiate(u[kk, :, i, :], self.spatial_grid[kk, :, 1])
        return u_derivs

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
        if self.include_bias:
            feature_names.append("1")
        for i, f in enumerate(self.functions):
            for c in self._combinations(
                self.n_input_features_, f.__code__.co_argcount, self.interaction_only
            ):
                feature_names.append(
                    self.function_names[i](*[input_features[j] for j in c])
                )
        for k in range(self.num_derivs):
            for j in range(self.n_input_features_):
                feature_names.append(
                    self.function_names[len(self.functions) + k](input_features[j])
                )
        for k in range(self.num_derivs):
            for i, f in enumerate(self.functions):
                for c in self._combinations(
                    self.n_input_features_,
                    f.__code__.co_argcount,
                    self.interaction_only,
                ):
                    for jj in range(self.n_input_features_):
                        feature_names.append(
                            self.function_names[i](*[input_features[j] for j in c])
                            + self.function_names[len(self.functions) + k](
                                input_features[jj]
                            )
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

        # Need to compute any derivatives now
        if self.spatial_grid is not None:
            # 1D space
            if len((self.spatial_grid).shape) == 1:
                num_gridpts = (self.spatial_grid).shape[0]
                num_time = n_samples // num_gridpts
                u = reshape(x, (num_gridpts, num_time, n_features))
                self.u_derivs = zeros(
                    (num_gridpts, num_time, n_features, self.num_derivs)
                )
                for i in range(num_time):
                    for j in range(self.num_derivs):
                        self.u_derivs[:, i, :, j] = FiniteDifference(
                            d=j + 1, is_uniform=self.is_uniform
                        )._differentiate(u[:, i, :], self.spatial_grid)
                if self.weak_form:
                    self.u_integrals = zeros((self.K, n_features, self.num_derivs))
                    for kk in range(n_features):
                        u_interp = RectBivariateSpline(
                            self.spatial_grid, self.temporal_grid, u[:, :, kk]
                        )
                        for k in range(self.K):
                            x1_k = self.domain_centers[k, 0] - self.Hx
                            x2_k = self.domain_centers[k, 0] + self.Hx
                            t1_k = self.domain_centers[k, -1] - self.Ht
                            t2_k = self.domain_centers[k, -1] + self.Ht
                            xgrid_k = linspace(x1_k, x2_k, self.num_pts_per_domain)
                            tgrid_k = linspace(t1_k, t2_k, self.num_pts_per_domain)
                            X, T = meshgrid(xgrid_k, tgrid_k, indexing="ij")
                            X = ravel(X)
                            T = ravel(T)
                            u_new = u_interp.ev(X, T)
                            u_new = reshape(
                                u_new,
                                (self.num_pts_per_domain, self.num_pts_per_domain, 1),
                            )
                            for j in range(self.num_derivs):
                                w_diff = self._smooth_ppoly(
                                    reshape(xgrid_k, (self.num_pts_per_domain, 1)),
                                    tgrid_k,
                                    k,
                                    j + 1,
                                    0,
                                    0,
                                )
                                self.u_integrals[k, kk, j] = (
                                    trapezoid(
                                        trapezoid(u_new * w_diff, x=xgrid_k, axis=0),
                                        x=tgrid_k,
                                        axis=0,
                                    )
                                    * (-1) ** (j + 1)
                                )

            # 2D space
            elif self.slen == 3:
                num_gridx = (self.spatial_grid).shape[0]
                num_gridy = (self.spatial_grid).shape[1]
                num_time = n_samples // num_gridx // num_gridy
                u = reshape(x, (num_gridx, num_gridy, num_time, n_features))
                self.u_derivs = self._make_2D_derivatives(u)
                deriv_list = asarray(
                    [
                        [1, 0],
                        [0, 1],
                        [2, 0],
                        [1, 1],
                        [0, 2],
                        [3, 0],
                        [2, 1],
                        [1, 2],
                        [0, 3],
                        [4, 0],
                        [3, 1],
                        [2, 2],
                        [1, 3],
                        [0, 4],
                    ]
                )
                signs_list = asarray([-1, -1, 1, 1, 1, -1, -1, -1, -1, 1, 1, 1, 1, 1])
                if self.weak_form:
                    self.u_integrals = zeros((self.K, n_features, self.num_derivs))
                    for kk in range(n_features):
                        u_interp = RegularGridInterpolator(
                            (
                                self.spatial_grid[:, 0, 0],
                                self.spatial_grid[0, :, 1],
                                self.temporal_grid,
                            ),
                            u[:, :, :, kk],
                        )
                        for k in range(self.K):
                            x1_k = self.domain_centers[k, 0] - self.Hx
                            x2_k = self.domain_centers[k, 0] + self.Hx
                            y1_k = self.domain_centers[k, 1] - self.Hy
                            y2_k = self.domain_centers[k, 1] + self.Hy
                            t1_k = self.domain_centers[k, -1] - self.Ht
                            t2_k = self.domain_centers[k, -1] + self.Ht
                            xgrid_k = linspace(x1_k, x2_k, self.num_pts_per_domain)
                            ygrid_k = linspace(y1_k, y2_k, self.num_pts_per_domain)
                            tgrid_k = linspace(t1_k, t2_k, self.num_pts_per_domain)
                            X, Y, T = meshgrid(xgrid_k, ygrid_k, tgrid_k, indexing="ij")
                            X = ravel(X)
                            Y = ravel(Y)
                            T = ravel(T)
                            u_new = u_interp((X, Y, T))
                            u_new = reshape(
                                u_new,
                                (
                                    self.num_pts_per_domain,
                                    self.num_pts_per_domain,
                                    self.num_pts_per_domain,
                                    1,
                                ),
                            )
                            for j in range(self.num_derivs):
                                w_diff = self._smooth_ppoly(
                                    transpose((xgrid_k, ygrid_k)),
                                    tgrid_k,
                                    k,
                                    deriv_list[j, 0],
                                    deriv_list[j, 1],
                                    0,
                                )
                                self.u_integrals[k, kk, j] = (
                                    trapezoid(
                                        trapezoid(
                                            trapezoid(
                                                u_new * w_diff, x=xgrid_k, axis=0
                                            ),
                                            x=ygrid_k,
                                            axis=0,
                                        ),
                                        x=tgrid_k,
                                        axis=0,
                                    )
                                    * signs_list[j]
                                )

        n_output_features = 0
        for f in self.functions:
            n_args = f.__code__.co_argcount
            n_output_features += len(
                list(self._combinations(n_features, n_args, self.interaction_only))
            )
        if self.spatial_grid is not None:
            # Add the mixed derivative library_terms
            n_output_features += n_output_features * n_features * self.num_derivs
            # Add the pure derivative library terms
            n_output_features += n_features * self.num_derivs
        if self.include_bias:
            n_output_features += 1
        self.n_output_features_ = n_output_features
        if self.function_names is None:
            self.function_names = list(
                map(
                    lambda i: (lambda *x: "f" + str(i) + "(" + ",".join(x) + ")"),
                    range(n_features),
                )
            )
        if self.spatial_grid is not None:
            if len((self.spatial_grid).shape) == 1:
                # pure derivative terms
                self.function_names = hstack(
                    (
                        self.function_names,
                        list(
                            map(
                                lambda i: (lambda *x: "".join(x) + "_" + "1" * i),
                                range(1, self.num_derivs + 1),
                            )
                        ),
                    )
                )
            elif self.slen == 3:
                deriv_strings = [
                    "1",
                    "2",
                    "11",
                    "12",
                    "22",
                    "111",
                    "112",
                    "122",
                    "222",
                    "1111",
                    "1112",
                    "1122",
                    "1222",
                    "2222",
                ]
                # pure derivative terms
                self.function_names = hstack(
                    (
                        self.function_names,
                        list(
                            map(
                                lambda i: (
                                    lambda *x: "".join(x) + "_" + deriv_strings[i]
                                ),
                                range(0, self.num_derivs),
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
            The matrix of features, where n_output_features is the number of
            features generated from applying the custom functions
            to the inputs.
        """
        check_is_fitted(self)

        x = check_array(x)

        n_samples, n_features = x.shape

        if n_features != self.n_input_features_:
            raise ValueError("x shape does not match training shape")

        if self.spatial_grid is not None:
            if self.slen == 1:
                num_gridpts = (self.spatial_grid).shape[0]
                num_time = n_samples // num_gridpts
            if self.slen == 3:
                num_gridx = (self.spatial_grid).shape[0]
                num_gridy = (self.spatial_grid).shape[1]
                num_time = n_samples // num_gridx // num_gridy
            if self.weak_form:
                n_samples = self.K

        xp = empty((n_samples, self.n_output_features_), dtype=x.dtype)
        library_idx = 0
        if self.include_bias:
            # To do: put in a bias term for the weak form
            xp[:, library_idx] = ones(n_samples)
            library_idx += 1
        if self.weak_form:
            if self.slen == 1:
                for f in self.functions:
                    for c in self._combinations(
                        self.n_input_features_,
                        f.__code__.co_argcount,
                        self.interaction_only,
                    ):
                        func = f(*[x[:, j] for j in c])
                        func = reshape(func, (num_gridpts, num_time))
                        func_final = zeros((self.K, 1))
                        func_interp = RectBivariateSpline(
                            self.spatial_grid, self.temporal_grid, func
                        )
                        for k in range(self.K):
                            x1_k = self.domain_centers[k, 0] - self.Hx
                            x2_k = self.domain_centers[k, 0] + self.Hx
                            t1_k = self.domain_centers[k, -1] - self.Ht
                            t2_k = self.domain_centers[k, -1] + self.Ht
                            xgrid_k = linspace(x1_k, x2_k, self.num_pts_per_domain)
                            tgrid_k = linspace(t1_k, t2_k, self.num_pts_per_domain)
                            X, T = meshgrid(xgrid_k, tgrid_k, indexing="ij")
                            X = ravel(X)
                            T = ravel(T)
                            func_new = func_interp.ev(X, T)
                            func_new = reshape(
                                func_new,
                                (self.num_pts_per_domain, self.num_pts_per_domain, 1),
                            )
                            w = self._smooth_ppoly(
                                reshape(xgrid_k, (self.num_pts_per_domain, 1)),
                                tgrid_k,
                                k,
                                0,
                                0,
                                0,
                            )
                            func_final[k] = trapezoid(
                                trapezoid(func_new * w, x=xgrid_k, axis=0),
                                x=tgrid_k,
                                axis=0,
                            )
                        xp[:, library_idx] = ravel(func_final)
                        library_idx += 1
            if self.slen == 3:
                for f in self.functions:
                    for c in self._combinations(
                        self.n_input_features_,
                        f.__code__.co_argcount,
                        self.interaction_only,
                    ):
                        func = f(*[x[:, j] for j in c])
                        func = reshape(func, (num_gridx, num_gridy, num_time))
                        func_final = zeros((self.K, 1))
                        func_interp = RegularGridInterpolator(
                            (
                                self.spatial_grid[:, 0, 0],
                                self.spatial_grid[0, :, 1],
                                self.temporal_grid,
                            ),
                            func,
                        )
                        for k in range(self.K):
                            x1_k = self.domain_centers[k, 0] - self.Hx
                            x2_k = self.domain_centers[k, 0] + self.Hx
                            y1_k = self.domain_centers[k, 1] - self.Hy
                            y2_k = self.domain_centers[k, 1] + self.Hy
                            t1_k = self.domain_centers[k, -1] - self.Ht
                            t2_k = self.domain_centers[k, -1] + self.Ht
                            xgrid_k = linspace(x1_k, x2_k, self.num_pts_per_domain)
                            ygrid_k = linspace(y1_k, y2_k, self.num_pts_per_domain)
                            tgrid_k = linspace(t1_k, t2_k, self.num_pts_per_domain)
                            X, Y, T = meshgrid(xgrid_k, ygrid_k, tgrid_k, indexing="ij")
                            X = ravel(X)
                            Y = ravel(Y)
                            T = ravel(T)
                            func_new = func_interp((X, Y, T))
                            func_new = reshape(
                                func_new,
                                (
                                    self.num_pts_per_domain,
                                    self.num_pts_per_domain,
                                    self.num_pts_per_domain,
                                    1,
                                ),
                            )
                            w = self._smooth_ppoly(
                                transpose((xgrid_k, ygrid_k)),
                                tgrid_k,
                                k,
                                0,
                                0,
                                0,
                            )
                            func_final[k] = trapezoid(
                                trapezoid(
                                    trapezoid(func_new * w, x=xgrid_k, axis=0),
                                    x=ygrid_k,
                                    axis=0,
                                ),
                                x=tgrid_k,
                                axis=0,
                            )
                        xp[:, library_idx] = ravel(func_final)
                        library_idx += 1
        else:
            for f in self.functions:
                for c in self._combinations(
                    self.n_input_features_,
                    f.__code__.co_argcount,
                    self.interaction_only,
                ):
                    xp[:, library_idx] = f(*[x[:, j] for j in c])
                    library_idx += 1

        # Need to recompute derivatives because the training
        # and testing data may have different number of time points
        # Need to compute any derivatives now
        if self.spatial_grid is not None:
            if self.slen == 1:
                u = reshape(x, (num_gridpts, num_time, n_features))
                u_derivs = zeros((num_gridpts, num_time, n_features, self.num_derivs))
                for i in range(num_time):
                    for j in range(self.num_derivs):
                        u_derivs[:, i, :, j] = FiniteDifference(
                            d=j + 1, is_uniform=self.is_uniform
                        )._differentiate(u[:, i, :], self.spatial_grid)
                u_derivs = asarray(
                    reshape(
                        u_derivs,
                        (num_gridpts * num_time, n_features, self.num_derivs),
                    )
                )
                if self.weak_form:
                    u_integrals = zeros((self.K, n_features, self.num_derivs))
                    for kk in range(n_features):
                        u_interp = RectBivariateSpline(
                            self.spatial_grid, self.temporal_grid, u[:, :, kk]
                        )
                        for k in range(self.K):
                            x1_k = self.domain_centers[k, 0] - self.Hx
                            x2_k = self.domain_centers[k, 0] + self.Hx
                            t1_k = self.domain_centers[k, -1] - self.Ht
                            t2_k = self.domain_centers[k, -1] + self.Ht
                            xgrid_k = linspace(x1_k, x2_k, self.num_pts_per_domain)
                            tgrid_k = linspace(t1_k, t2_k, self.num_pts_per_domain)
                            X, T = meshgrid(xgrid_k, tgrid_k, indexing="ij")
                            X = ravel(X)
                            T = ravel(T)
                            u_new = u_interp.ev(X, T)
                            u_new = reshape(
                                u_new,
                                (self.num_pts_per_domain, self.num_pts_per_domain, 1),
                            )
                            for j in range(self.num_derivs):
                                w_diff = self._smooth_ppoly(
                                    reshape(xgrid_k, (self.num_pts_per_domain, 1)),
                                    tgrid_k,
                                    k,
                                    j + 1,
                                    0,
                                    0,
                                )
                                u_integrals[k, kk, j] = (
                                    trapezoid(
                                        trapezoid(u_new * w_diff, x=xgrid_k, axis=0),
                                        x=tgrid_k,
                                        axis=0,
                                    )
                                    * (-1) ** (j + 1)
                                )

            elif self.slen == 3:
                u = reshape(x, (num_gridx, num_gridy, num_time, n_features))
                u_derivs = self._make_2D_derivatives(u)
                u_derivs = asarray(
                    reshape(
                        u_derivs,
                        (num_gridx * num_gridy * num_time, n_features, self.num_derivs),
                    )
                )
                if self.weak_form:
                    deriv_list = asarray(
                        [
                            [1, 0],
                            [0, 1],
                            [2, 0],
                            [1, 1],
                            [0, 2],
                            [3, 0],
                            [2, 1],
                            [1, 2],
                            [0, 3],
                            [4, 0],
                            [3, 1],
                            [2, 2],
                            [1, 3],
                            [0, 4],
                        ]
                    )
                    signs_list = asarray(
                        [-1, -1, 1, 1, 1, -1, -1, -1, -1, 1, 1, 1, 1, 1]
                    )
                    u_integrals = zeros((self.K, n_features, self.num_derivs))
                    for kk in range(n_features):
                        u_interp = RegularGridInterpolator(
                            (
                                self.spatial_grid[:, 0, 0],
                                self.spatial_grid[0, :, 1],
                                self.temporal_grid,
                            ),
                            u[:, :, :, kk],
                        )
                        for k in range(self.K):
                            x1_k = self.domain_centers[k, 0] - self.Hx
                            x2_k = self.domain_centers[k, 0] + self.Hx
                            y1_k = self.domain_centers[k, 1] - self.Hy
                            y2_k = self.domain_centers[k, 1] + self.Hy
                            t1_k = self.domain_centers[k, -1] - self.Ht
                            t2_k = self.domain_centers[k, -1] + self.Ht
                            xgrid_k = linspace(x1_k, x2_k, self.num_pts_per_domain)
                            ygrid_k = linspace(y1_k, y2_k, self.num_pts_per_domain)
                            tgrid_k = linspace(t1_k, t2_k, self.num_pts_per_domain)
                            X, Y, T = meshgrid(xgrid_k, ygrid_k, tgrid_k, indexing="ij")
                            X = ravel(X)
                            Y = ravel(Y)
                            T = ravel(T)
                            u_new = u_interp((X, Y, T))
                            u_new = reshape(
                                u_new,
                                (
                                    self.num_pts_per_domain,
                                    self.num_pts_per_domain,
                                    self.num_pts_per_domain,
                                    1,
                                ),
                            )
                            for j in range(self.num_derivs):
                                w_diff = self._smooth_ppoly(
                                    transpose((xgrid_k, ygrid_k)),
                                    tgrid_k,
                                    k,
                                    deriv_list[j, 0],
                                    deriv_list[j, 1],
                                    0,
                                )
                                u_integrals[k, kk, j] = (
                                    trapezoid(
                                        trapezoid(
                                            trapezoid(
                                                u_new * w_diff, x=xgrid_k, axis=0
                                            ),
                                            x=ygrid_k,
                                            axis=0,
                                        ),
                                        x=tgrid_k,
                                        axis=0,
                                    )
                                    * signs_list[j]
                                )

            def identity_function(y):
                return y

            if self.weak_form:
                for k in range(self.num_derivs):
                    for kk in range(n_features):
                        xp[:, library_idx] = identity_function(u_integrals[:, kk, k])
                        library_idx += 1
            else:
                for k in range(self.num_derivs):
                    for kk in range(n_features):
                        xp[:, library_idx] = identity_function(u_derivs[:, kk, k])
                        library_idx += 1
            if self.slen == 1:
                # All the mixed derivative/non-derivative terms
                for d in range(self.num_derivs):
                    for kk in range(n_features):
                        if self.weak_form:
                            for f in self.functions:
                                for c in self._combinations(
                                    self.n_input_features_,
                                    f.__code__.co_argcount,
                                    self.interaction_only,
                                ):
                                    func = f(*[x[:, j] for j in c])
                                    if d == 0 or d == 1:
                                        deriv_term = identity_function(
                                            u_derivs[:, kk, 0]
                                        )
                                    elif d == 2 or d == 3:
                                        deriv_term = identity_function(
                                            u_derivs[:, kk, 1]
                                        )
                                    func = reshape(func, (num_gridpts, num_time))
                                    deriv_term = reshape(
                                        deriv_term, (num_gridpts, num_time)
                                    )
                                    func_final = zeros((self.K, 1))
                                    func_interp = RectBivariateSpline(
                                        self.spatial_grid, self.temporal_grid, func
                                    )
                                    deriv_interp = RectBivariateSpline(
                                        self.spatial_grid,
                                        self.temporal_grid,
                                        deriv_term,
                                    )
                                    for k in range(self.K):
                                        x1_k = self.domain_centers[k, 0] - self.Hx
                                        x2_k = self.domain_centers[k, 0] + self.Hx
                                        t1_k = self.domain_centers[k, -1] - self.Ht
                                        t2_k = self.domain_centers[k, -1] + self.Ht
                                        xgrid_k = linspace(
                                            x1_k, x2_k, self.num_pts_per_domain
                                        )
                                        tgrid_k = linspace(
                                            t1_k, t2_k, self.num_pts_per_domain
                                        )
                                        X, T = meshgrid(xgrid_k, tgrid_k, indexing="ij")
                                        X = ravel(X)
                                        T = ravel(T)
                                        func_new = func_interp.ev(X, T)
                                        deriv_new = deriv_interp.ev(X, T)
                                        func_new = reshape(
                                            func_new,
                                            (
                                                self.num_pts_per_domain,
                                                self.num_pts_per_domain,
                                                1,
                                            ),
                                        )
                                        deriv_new = reshape(
                                            deriv_new,
                                            (
                                                self.num_pts_per_domain,
                                                self.num_pts_per_domain,
                                                1,
                                            ),
                                        )
                                        w = self._smooth_ppoly(
                                            reshape(
                                                xgrid_k, (self.num_pts_per_domain, 1)
                                            ),
                                            tgrid_k,
                                            k,
                                            0,
                                            0,
                                            0,
                                        )
                                        if d == 0:
                                            w_func = func_new * w
                                        if d == 1 or d == 2:
                                            w_func = (-1) * FiniteDifference(
                                                d=1, is_uniform=self.is_uniform
                                            )._differentiate(
                                                func_new * w,
                                                xgrid_k,
                                            )
                                        if d == 3:
                                            w_func = FiniteDifference(
                                                d=2, is_uniform=self.is_uniform
                                            )._differentiate(
                                                func_new * w,
                                                xgrid_k,
                                            )
                                        func_final[k] = trapezoid(
                                            trapezoid(
                                                w_func * deriv_new,
                                                x=xgrid_k,
                                                axis=0,
                                            ),
                                            x=tgrid_k,
                                            axis=0,
                                        )
                                    xp[:, library_idx] = ravel(func_final)
                                    library_idx += 1
                        else:
                            for f in self.functions:
                                for c in self._combinations(
                                    self.n_input_features_,
                                    f.__code__.co_argcount,
                                    self.interaction_only,
                                ):
                                    xp[:, library_idx] = f(
                                        *[x[:, j] for j in c]
                                    ) * identity_function(u_derivs[:, kk, d])
                                    library_idx += 1

            if self.slen == 3:
                # All the mixed derivative/non-derivative terms
                # Note this is really annoying to integrate these mixed terms
                # by parts in 2D so we don't do it for now.
                for d in range(self.num_derivs):
                    for kk in range(n_features):
                        if self.weak_form:
                            for f in self.functions:
                                for c in self._combinations(
                                    self.n_input_features_,
                                    f.__code__.co_argcount,
                                    self.interaction_only,
                                ):
                                    func = f(*[x[:, j] for j in c])
                                    deriv_term = identity_function(u_derivs[:, kk, d])

                                    func = reshape(
                                        func, (num_gridx, num_gridy, num_time)
                                    )
                                    deriv_term = reshape(
                                        deriv_term, (num_gridx, num_gridy, num_time)
                                    )
                                    func_final = zeros((self.K, 1))
                                    func_interp = RegularGridInterpolator(
                                        (
                                            self.spatial_grid[:, 0, 0],
                                            self.spatial_grid[0, :, 1],
                                            self.temporal_grid,
                                        ),
                                        func,
                                    )
                                    deriv_interp = RegularGridInterpolator(
                                        (
                                            self.spatial_grid[:, 0, 0],
                                            self.spatial_grid[0, :, 1],
                                            self.temporal_grid,
                                        ),
                                        deriv_term,
                                    )
                                    for k in range(self.K):
                                        x1_k = self.domain_centers[k, 0] - self.Hx
                                        x2_k = self.domain_centers[k, 0] + self.Hx
                                        y1_k = self.domain_centers[k, 1] - self.Hy
                                        y2_k = self.domain_centers[k, 1] + self.Hy
                                        t1_k = self.domain_centers[k, -1] - self.Ht
                                        t2_k = self.domain_centers[k, -1] + self.Ht
                                        xgrid_k = linspace(
                                            x1_k, x2_k, self.num_pts_per_domain
                                        )
                                        ygrid_k = linspace(
                                            y1_k, y2_k, self.num_pts_per_domain
                                        )
                                        tgrid_k = linspace(
                                            t1_k, t2_k, self.num_pts_per_domain
                                        )
                                        X, Y, T = meshgrid(
                                            xgrid_k, ygrid_k, tgrid_k, indexing="ij"
                                        )
                                        X = ravel(X)
                                        Y = ravel(Y)
                                        T = ravel(T)
                                        func_new = func_interp((X, Y, T))
                                        deriv_new = deriv_interp((X, Y, T))
                                        func_new = reshape(
                                            func_new,
                                            (
                                                self.num_pts_per_domain,
                                                self.num_pts_per_domain,
                                                self.num_pts_per_domain,
                                                1,
                                            ),
                                        )
                                        deriv_new = reshape(
                                            deriv_new,
                                            (
                                                self.num_pts_per_domain,
                                                self.num_pts_per_domain,
                                                self.num_pts_per_domain,
                                                1,
                                            ),
                                        )
                                        w = self._smooth_ppoly(
                                            transpose((xgrid_k, ygrid_k)),
                                            tgrid_k,
                                            k,
                                            0,
                                            0,
                                            0,
                                        )
                                        func_final[k] = trapezoid(
                                            trapezoid(
                                                trapezoid(
                                                    w * func_new * deriv_new,
                                                    x=xgrid_k,
                                                    axis=0,
                                                ),
                                                x=ygrid_k,
                                                axis=0,
                                            ),
                                            x=tgrid_k,
                                            axis=0,
                                        )
                                    xp[:, library_idx] = ravel(func_final)
                                    library_idx += 1
                        else:
                            for f in self.functions:
                                for c in self._combinations(
                                    self.n_input_features_,
                                    f.__code__.co_argcount,
                                    self.interaction_only,
                                ):
                                    xp[:, library_idx] = f(
                                        *[x[:, j] for j in c]
                                    ) * identity_function(u_derivs[:, kk, d])
                                    library_idx += 1
        # If library bagging, return xp missing a single column
        if self.library_ensemble:
            inds = range(self.n_output_features_)
            inds = delete(inds, self.ensemble_indices)
            return xp[:, inds]
        else:
            return xp
