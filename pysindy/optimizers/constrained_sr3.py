import warnings

import cvxpy as cp
import numpy as np
from scipy.linalg import cho_factor
from sklearn.exceptions import ConvergenceWarning

from ..utils import get_regularization
from ..utils import reorder_constraints
from .sr3 import SR3


class ConstrainedSR3(SR3):
    """
    Sparse relaxed regularized regression with linear equality constraints.

    Attempts to minimize the objective function

    .. math::

        0.5\\|y-Xw\\|^2_2 + \\lambda \\times R(v)
        + (0.5 / nu)\\|w-v\\|^2_2

        \\text{subject to}

    .. math::

        Cw = d

    over v and w where :math:`R(v)` is a regularization function, C is a
    constraint matrix, and d is a vector of values. See the following
    reference for more details:

        Champion, Kathleen, et al. "A unified sparse optimization framework
        to learn parsimonious physics-informed models from data."
        arXiv preprint arXiv:1906.10612 (2019).

    Parameters
    ----------
    threshold : float, optional (default 0.1)
        Determines the strength of the regularization. When the
        regularization function R is the l0 norm, the regularization
        is equivalent to performing hard thresholding, and lambda
        is chosen to threshold at the value given by this parameter.
        This is equivalent to choosing lambda = threshold^2 / (2 * nu).

    nu : float, optional (default 1)
        Determines the level of relaxation. Decreasing nu encourages
        w and v to be close, whereas increasing nu allows the
        regularized coefficients v to be farther from w.

    tol : float, optional (default 1e-5)
        Tolerance used for determining convergence of the optimization
        algorithm.

    thresholder : string, optional (default 'l0')
        Regularization function to use. Currently implemented options
        are 'l0' (l0 norm), 'l1' (l1 norm), 'cad' (clipped
        absolute deviation), 'weighted_l0' (weighted l0 norm), and
        'weighted_l1' (weighted l1 norm).

    max_iter : int, optional (default 30)
        Maximum iterations of the optimization algorithm.

    fit_intercept : boolean, optional (default False)
        Whether to calculate the intercept for this model. If set to false, no
        intercept will be used in calculations.

    constraint_lhs : numpy ndarray, shape (n_constraints, n_features * n_targets), \
            optional (default None)
        The left hand side matrix C of Cw <= d.
        There should be one row per constraint.

    constraint_rhs : numpy ndarray, shape (n_constraints,), optional (default None)
        The right hand side vector d of Cw <= d.

    constraint_order : string, optional (default "target")
        The format in which the constraints ``constraint_lhs`` were passed.
        Must be one of "target" or "feature".
        "target" indicates that the constraints are grouped by target:
        i.e. the first ``n_features`` columns
        correspond to constraint coefficients on the library features for the first
        target (variable), the next ``n_features`` columns to the library features
        for the second target (variable), and so on.
        "feature" indicates that the constraints are grouped by library feature:
        the first ``n_targets`` columns correspond to the first library feature,
        the next ``n_targets`` columns to the second library feature, and so on.

    normalize : boolean, optional (default False)
        This parameter is ignored when fit_intercept is set to False. If True,
        the regressors X will be normalized before regression by subtracting
        the mean and dividing by the l2-norm.

    normalize_columns : boolean, optional (default False)
        Normalize the columns of x (the SINDy library terms). Note that
        this parameter is incompatible with the constraints!

    copy_X : boolean, optional (default True)
        If True, X will be copied; else, it may be overwritten.

    initial_guess : np.ndarray, shape (n_features) or (n_targets, n_features), \
                optional (default None)
        Initial guess for coefficients ``coef_``, (v in the mathematical equations)
        If None, least-squares is used to obtain an initial guess.

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

    inequality_constraints : bool, optional
        If True, CVXPY methods are used to solve the problem.

    Attributes
    ----------
    coef_ : array, shape (n_features,) or (n_targets, n_features)
        Regularized weight vector(s). This is the v in the objective
        function.

    coef_full_ : array, shape (n_features,) or (n_targets, n_features)
        Weight vector(s) that are not subjected to the regularization.
        This is the w in the objective function.

    unbias : boolean
        Whether to perform an extra step of unregularized linear regression
        to unbias the coefficients for the identified support.
        ``unbias`` is automatically set to False if a constraint is used and
        is otherwise left uninitialized.
    """

    def __init__(
        self,
        threshold=0.1,
        nu=1.0,
        tol=1e-5,
        thresholder="l0",
        max_iter=30,
        trimming_fraction=0.0,
        trimming_step_size=1.0,
        constraint_lhs=None,
        constraint_rhs=None,
        constraint_order="target",
        normalize_columns=False,
        normalize=False,
        fit_intercept=False,
        copy_X=True,
        initial_guess=None,
        thresholds=None,
        inequality_constraints=False,
    ):
        super(ConstrainedSR3, self).__init__(
            threshold=threshold,
            nu=nu,
            tol=tol,
            thresholder=thresholder,
            thresholds=thresholds,
            trimming_fraction=trimming_fraction,
            trimming_step_size=trimming_step_size,
            max_iter=max_iter,
            initial_guess=initial_guess,
            normalize=normalize,
            fit_intercept=fit_intercept,
            copy_X=copy_X,
            normalize_columns=normalize_columns,
        )

        self.reg = get_regularization(thresholder)
        self.use_constraints = (constraint_lhs is not None) and (
            constraint_rhs is not None
        )

        if self.use_constraints:
            if constraint_order not in ("feature", "target"):
                raise ValueError(
                    "constraint_order must be either 'feature' or 'target'"
                )

            self.constraint_lhs = constraint_lhs
            self.constraint_rhs = constraint_rhs
            self.unbias = False
            self.constraint_order = constraint_order

        if inequality_constraints and not self.use_constraints:
            raise ValueError(
                "Use of inequality constraints requires constraint_lhs and "
                "constraint_rhs."
            )

        if inequality_constraints and (
            thresholder.lower() != "l1"
            and thresholder.lower() != "l2"
            and thresholder.lower() != "weighted_l1"
            and thresholder.lower() != "weighted_l2"
        ):
            raise ValueError(
                "Use of inequality constraints requires a convex regularizer."
            )
        self.inequality_constraints = inequality_constraints

    def _set_threshold(self, threshold):
        self.threshold = threshold

    def _update_full_coef_constraints(self, H, x_transpose_y, coef_sparse):
        g = x_transpose_y + coef_sparse / self.nu
        inv1 = np.linalg.inv(H)
        inv1_mod = np.kron(inv1, np.eye(coef_sparse.shape[1]))
        inv2 = np.linalg.inv(
            self.constraint_lhs.dot(inv1_mod).dot(self.constraint_lhs.T)
        )

        rhs = g.flatten() + self.constraint_lhs.T.dot(inv2).dot(
            self.constraint_rhs - self.constraint_lhs.dot(inv1_mod).dot(g.flatten())
        )
        rhs = rhs.reshape(g.shape)
        return inv1.dot(rhs)

    def _update_coef_cvxpy(self, x, y, coef_sparse):
        xi = cp.Variable(coef_sparse.shape[0] * coef_sparse.shape[1])
        cost = cp.sum_squares(x @ xi - y.flatten())
        if self.thresholder.lower() == "l1":
            cost = cost + self.threshold * cp.norm1(xi)
        elif self.thresholder.lower() == "weighted_l1":
            cost = cost + cp.norm1(self.threshold * xi)
        elif self.thresholder.lower() == "l2":
            cost = cost + self.threshold * cp.norm2(xi)
        elif self.thresholder.lower() == "weighted_l2":
            cost = cost + cp.norm2(self.thresholds * xi)
        if self.use_constraints:
            if self.inequality_constraints:
                prob = cp.Problem(
                    cp.Minimize(cost),
                    [self.constraint_lhs @ xi <= self.constraint_rhs],
                )
            else:
                prob = cp.Problem(
                    cp.Minimize(cost),
                    [self.constraint_lhs @ xi == self.constraint_rhs],
                )
        else:
            prob = cp.Problem(cp.Minimize(cost))
        prob.solve(max_iter=50000, eps_abs=self.tol, eps_rel=self.tol)
        if xi.value is None:
            warnings.warn(
                "Infeasible solve, probably an issue with the regularizer "
                " or the constraint that was used.",
                ConvergenceWarning,
            )
            return None
        coef_new = (xi.value).reshape(coef_sparse.shape)
        return coef_new

    def _update_sparse_coef(self, coef_full):
        """Update the regularized weight vector"""
        if self.thresholds is None:
            return super(ConstrainedSR3, self)._update_sparse_coef(coef_full)
        else:
            coef_sparse = self.prox(coef_full, self.thresholds.T)
        self.history_.append(coef_sparse.T)
        return coef_sparse

    def _objective(self, x, y, coef_full, coef_sparse, trimming_array=None):
        """Objective function"""
        R2 = (y - np.dot(x, coef_full)) ** 2
        D2 = (coef_full - coef_sparse) ** 2
        if self.use_trimming:
            assert trimming_array is not None
            R2 *= trimming_array.reshape(x.shape[0], 1)
        if self.thresholds is None:
            return (
                0.5 * np.sum(R2)
                + self.reg(coef_full, 0.5 * self.threshold ** 2 / self.nu)
                + 0.5 * np.sum(D2) / self.nu
            )
        else:
            return (
                0.5 * np.sum(R2)
                + self.reg(coef_full, 0.5 * self.thresholds.T ** 2 / self.nu)
                + 0.5 * np.sum(D2) / self.nu
            )

    def _reduce(self, x, y):
        """
        Perform at most ``self.max_iter`` iterations of the SR3 algorithm
        with inequality constraints.

        Assumes initial guess for coefficients is stored in ``self.coef_``.
        """
        coef_sparse = self.coef_.T
        coef_full = coef_sparse.copy()
        n_samples, n_features = x.shape
        n_targets = y.shape[1]

        self.Theta = x
        x_normed = np.copy(x)
        if self.normalize_columns:
            reg = np.zeros(n_features)
            for i in range(n_features):
                reg[i] = 1.0 / np.linalg.norm(x[:, i], 2)
                x_normed[:, i] = reg[i] * x[:, i]
                coef_sparse[i, :] = coef_sparse[i, :] / reg[i]

        if self.use_trimming:
            trimming_array = np.repeat(1.0 - self.trimming_fraction, n_samples)
            self.history_trimming_ = [trimming_array]

        if self.use_constraints and self.constraint_order.lower() == "target":
            self.constraint_lhs = reorder_constraints(self.constraint_lhs, n_features)

        # Precompute some objects for upcoming least-squares solves.
        # Assumes that self.nu is fixed throughout optimization procedure.
        H = np.dot(x_normed.T, x_normed) + np.diag(
            np.full(x_normed.shape[1], 1.0 / self.nu)
        )
        x_transpose_y = np.dot(x_normed.T, y)
        if not self.use_constraints:
            cho = cho_factor(H)
        if self.inequality_constraints:
            # Precompute some objects for optimization
            x_expanded = np.zeros((n_samples, n_targets, n_features, n_targets))
            for i in range(n_targets):
                x_expanded[:, i, :, i] = x_normed
            x_expanded = np.reshape(
                x_expanded, (n_samples * n_targets, n_targets * n_features)
            )

        objective_history = []
        if self.inequality_constraints:
            coef_sparse = self._update_coef_cvxpy(x_expanded, y, coef_sparse)
            objective_history.append(
                self._objective(x_normed, y, coef_full, coef_sparse)
            )
        else:
            for _ in range(self.max_iter):
                if self.use_trimming:
                    x_weighted = x_normed * trimming_array.reshape(n_samples, 1)
                    H = np.dot(x_weighted.T, x_normed) + np.diag(
                        np.full(x_normed.shape[1], 1.0 / self.nu)
                    )
                    x_transpose_y = np.dot(x_weighted.T, y)
                    if not self.use_constraints:
                        cho = cho_factor(H)
                    trimming_grad = 0.5 * np.sum(
                        (y - x_normed.dot(coef_full)) ** 2, axis=1
                    )
                if self.use_constraints:
                    coef_full = self._update_full_coef_constraints(
                        H, x_transpose_y, coef_sparse
                    )
                else:
                    coef_full = self._update_full_coef(cho, x_transpose_y, coef_sparse)
                coef_sparse = self._update_sparse_coef(coef_full)

                if self.use_trimming:
                    trimming_array = self._update_trimming_array(
                        coef_full, trimming_array, trimming_grad
                    )

                    objective_history.append(
                        self._objective(
                            x_normed, y, coef_full, coef_sparse, trimming_array
                        )
                    )
                else:
                    objective_history.append(
                        self._objective(x_normed, y, coef_full, coef_sparse)
                    )
                if self._convergence_criterion() < self.tol:
                    # TODO: Update this for trimming/constraints
                    break
            else:
                warnings.warn(
                    "SR3._reduce did not converge after {} iterations.".format(
                        self.max_iter
                    ),
                    ConvergenceWarning,
                )

        if self.use_constraints and self.constraint_order.lower() == "target":
            self.constraint_lhs = reorder_constraints(
                self.constraint_lhs, n_features, output_order="target"
            )
        if self.normalize_columns:
            self.coef_ = np.multiply(reg, coef_sparse.T)
            self.coef_full_ = np.multiply(reg, coef_full.T)
        else:
            self.coef_ = coef_sparse.T
            self.coef_full_ = coef_full.T
        if self.use_trimming:
            self.trimming_array = trimming_array
        self.objective_history = objective_history
