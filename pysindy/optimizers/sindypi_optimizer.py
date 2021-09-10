import warnings

import cvxpy as cp
import numpy as np
from sklearn.exceptions import ConvergenceWarning

from .sr3 import SR3


class SINDyPI(SR3):
    """
    SINDy-PI optimizer

    Attempts to minimize the objective function

    .. math::

        0.5\\|X-Xw\\|^2_2 + \\lambda \\times R(v)

    over w where :math:`R(v)` is a regularization function. See the following
    reference for more details:

        Kaheman, Kadierdan, J. Nathan Kutz, and Steven L. Brunton. SINDy-PI:
        a robust algorithm for parallel implicit sparse identification of
        nonlinear dynamics.
        Proceedings of the Royal Society A 476.2242 (2020): 20200279.

    Parameters
    ----------
    threshold : float, optional (default 0.1)
        Determines the strength of the regularization. When the
        regularization function R is the l0 norm, the regularization
        is equivalent to performing hard thresholding, and lambda
        is chosen to threshold at the value given by this parameter.
        This is equivalent to choosing lambda = threshold^2 / (2 * nu).

    tol : float, optional (default 1e-5)
        Tolerance used for determining convergence of the optimization
        algorithm.

    thresholder : string, optional (default 'l1')
        Regularization function to use. Currently implemented options
        are 'l1' (l1 norm), 'weighted_l1' (weighted l1 norm), l2, and
        'weighted_l2' (weighted l2 norm)

    max_iter : int, optional (default 10000)
        Maximum iterations of the optimization algorithm.

    fit_intercept : boolean, optional (default False)
        Whether to calculate the intercept for this model. If set to false, no
        intercept will be used in calculations.

    normalize : boolean, optional (default False)
        This parameter is ignored when fit_intercept is set to False. If True,
        the regressors X will be normalized before regression by subtracting
        the mean and dividing by the l2-norm.

    normalize_columns : boolean, optional (default False)
        This parameter normalizes the columns of Theta before the
        optimization is done. This tends to standardize the columns
        to similar magnitudes, often improving performance.

    copy_X : boolean, optional (default True)
        If True, X will be copied; else, it may be overwritten.

    thresholds : np.ndarray, shape (n_targets, n_features), optional \
            (default None)
        Array of thresholds for each library function coefficient.
        Each row corresponds to a measurement variable and each column
        to a function from the feature library.
        Recall that SINDy seeks a matrix :math:`\\Xi` such that
        :math:`\\dot{X} \\approx \\Theta(X)\\Xi`.
        ``thresholds[i, j]`` should specify the threshold to be used for the
        (j + 1, i + 1) entry of :math:`\\Xi`. That is to say it should give the
        threshold to be used for the (j + 1)st library function in the equation
        for the (i + 1)st measurement variable.

    model_subset : np.ndarray, shape(n_models), optional (default None)
        List of indices to compute models for. If list is not provided,
        the default is to compute SINDy-PI models for all possible
        candidate functions. This can take a long time for 4D systems
        or larger.

    Attributes
    ----------
    coef_ : array, shape (n_features,) or (n_targets, n_features)
        Regularized weight vector(s). This is the v in the objective
        function.

    unbias : boolean
        Whether to perform an extra step of unregularized linear regression
        to unbias the coefficients for the identified support.
        ``unbias`` is automatically set to False if a constraint is used and
        is otherwise left uninitialized.

    Theta : np.ndarray, shape (n_samples, n_features)
        The Theta matrix to be used in the optimization. We save it as
        an attribute because access to the full library of terms is needed for
        the SINDy-PI ODE after the optimization.
    """

    def __init__(
        self,
        threshold=0.1,
        tol=1e-5,
        thresholder="l1",
        max_iter=10000,
        normalize=False,
        fit_intercept=False,
        copy_X=True,
        thresholds=None,
        model_subset=None,
        normalize_columns=False,
    ):
        super(SINDyPI, self).__init__(
            threshold=threshold,
            tol=tol,
            thresholder=thresholder,
            max_iter=max_iter,
            normalize=normalize,
            fit_intercept=fit_intercept,
            copy_X=copy_X,
            normalize_columns=normalize_columns,
        )

        if max_iter <= 0:
            raise ValueError("max_iter must be positive")
        if threshold < 0.0:
            raise ValueError("threshold must not be negative")
        if (
            thresholder.lower() != "l1"
            and thresholder.lower() != "l2"
            and thresholder.lower() != "weighted_l1"
            and thresholder.lower() != "weighted_l2"
        ):
            raise ValueError(
                "l0 and other nonconvex regularizers are not implemented "
                " in current version of SINDy-PI"
            )
        if thresholder[:8].lower() == "weighted" and thresholds is None:
            raise ValueError("weighted thresholder requires the thresholds parameter")
        if thresholder[:8].lower() != "weighted" and thresholds is not None:
            raise ValueError(
                "The thresholds argument cannot be used without a weighted"
                " thresholder, e.g. thresholder='weighted_l0'"
            )
        if thresholds is not None and np.any(thresholds < 0):
            raise ValueError("thresholds cannot contain negative entries")

        self.thresholds = thresholds
        self.unbias = False
        self.Theta = None
        self.model_subset = model_subset

    def _set_threshold(self, threshold):
        self.threshold = threshold

    def _update_full_coef_constraints(self, x):
        N = x.shape[1]
        xi = cp.Variable((N, N))
        if (self.thresholder).lower() in ("l1", "weighted_l1"):
            if self.thresholds is None:
                cost = cp.sum_squares(x - x @ xi) + self.threshold * cp.norm1(xi)
            else:
                cost = cp.sum_squares(x - x @ xi) + cp.norm1(self.thresholds @ xi)
        if (self.thresholder).lower() in ("l2", "weighted_l2"):
            if self.thresholds is None:
                cost = cp.sum_squares(x - x @ xi) + self.threshold * cp.norm2(xi) ** 2
            else:
                cost = cp.sum_squares(x - x @ xi) + cp.norm2(self.thresholds @ xi) ** 2
        prob = cp.Problem(
            cp.Minimize(cost),
            [cp.diag(xi) == np.zeros(N)],
        )

        # Beware: hard-coding the tolerances
        prob.solve(
            max_iter=self.max_iter, verbose=True, eps_abs=self.tol, eps_rel=self.tol
        )

        if xi.value is None:
            warnings.warn(
                "Infeasible solve, try changing your library",
                ConvergenceWarning,
            )
        return xi.value

    def _update_parallel_coef_constraints(self, x):
        N = x.shape[1]
        xi_final = np.zeros((N, N))

        # Todo: parallelize this for loop with Multiprocessing/joblib
        # for i in range(N):
        if self.model_subset is None:
            self.model_subset = range(N)
        for i in self.model_subset:
            xi = cp.Variable(N)
            # Note that norm choice below must be convex,
            # so thresholder must be L1 or L2
            if (self.thresholder).lower() in ("l1", "weighted_l1"):
                if self.thresholds is None:
                    cost = cp.sum_squares(x[:, i] - x @ xi) + self.threshold * cp.norm1(
                        xi
                    )
                else:
                    cost = cp.sum_squares(x[:, i] - x @ xi) + cp.norm1(
                        self.thresholds[i, :] @ xi
                    )
            if (self.thresholder).lower() in ("l2", "weighted_l2"):
                if self.thresholds is None:
                    cost = (
                        cp.sum_squares(x[:, i] - x @ xi)
                        + self.threshold * cp.norm2(xi) ** 2
                    )
                else:
                    cost = (
                        cp.sum_squares(x[:, i] - x @ xi)
                        + cp.norm2(self.thresholds[i, :] @ xi) ** 2
                    )
            prob = cp.Problem(
                cp.Minimize(cost),
                [xi[i] == 0.0],
            )
            try:
                prob.solve(max_iter=self.max_iter, eps_abs=self.tol, eps_rel=self.tol)
                if xi.value is None:
                    warnings.warn(
                        "Infeasible solve on iteration " + str(i) + ", try "
                        "changing your library",
                        ConvergenceWarning,
                    )
                xi_final[:, i] = xi.value
            # Annoying error coming from L2 norm switching to use the ECOS
            # solver, which uses "max_iters" instead of "max_iter", and
            # similar semantic changes for the other variables.
            except TypeError:
                prob.solve(max_iters=self.max_iter, abstol=self.tol, reltol=self.tol)
                if xi.value is None:
                    warnings.warn(
                        "Infeasible solve on iteration " + str(i) + ", try "
                        "changing your library",
                        ConvergenceWarning,
                    )
                xi_final[:, i] = xi.value
            except cp.error.SolverError:
                print("Solver failed on model ", str(i), ", setting coefs to zeros")
                xi_final[:, i] = np.zeros(N)
        return xi_final

    def _reduce(self, x, y):
        """
        Perform at most ``self.max_iter`` iterations of the SINDy-PI
        optimization problem, using CVXPY.
        """
        self.Theta = x  # Need to save the library for post-processing
        n_samples, n_features = x.shape
        x_normed = np.copy(x)
        if self.normalize_columns:
            reg = np.zeros(n_features)
            for i in range(n_features):
                reg[i] = 1.0 / np.linalg.norm(x[:, i], 2)
                x_normed[:, i] = reg[i] * x[:, i]
        # coef = self._update_full_coef_constraints(x_normed)
        coef = self._update_parallel_coef_constraints(x_normed)
        if self.normalize_columns:
            self.coef_ = np.multiply(reg, coef.T)
        else:
            self.coef_ = coef.T
