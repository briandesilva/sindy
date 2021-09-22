import numpy as np

from .base import BaseOptimizer


class FROLS(BaseOptimizer):
    """Sequentially thresholded least squares algorithm.

    Attempts to minimize the objective function
    :math:`\\|y - Xw\\|^2_2 + \\alpha \\|w\\|^2_2`
    by iteractively selecting the most correlated
    function in the library. This is a greedy algorithm.

    See the following reference for more details:

        Billings, Stephen A. Nonlinear system identification:
        NARMAX methods in the time, frequency, and spatio-temporal domains.
        John Wiley & Sons, 2013.

    Parameters
    ----------
    fit_intercept : boolean, optional (default False)
        Whether to calculate the intercept for this model. If set to false, no
        intercept will be used in calculations.

    normalize : boolean, optional (default False)
        This parameter is ignored when fit_intercept is set to False. If True,
        the regressors X will be normalized before regression by subtracting
        the mean and dividing by the l2-norm.

    normalize_columns : boolean, optional (default False)
        Normalize the columns of x (the SINDy library terms)

    copy_X : boolean, optional (default True)
        If True, X will be copied; else, it may be overwritten.

    L0_penalty : float, optional (default None)
        If passed, compute the MSE errors with an extra L0 term with
        strength equal to L0_penalty times the condition number of Theta.

    max_iter : int, optional (default 10)
        Maximum iterations of the optimization algorithm. This determines
        the number of nonzero terms chosen by the FROLS algorithm.

    Attributes
    ----------
    coef_ : array, shape (n_features,) or (n_targets, n_features)
        Weight vector(s).

    ind_ : array, shape (n_features,) or (n_targets, n_features)
        Array of 0s and 1s indicating which coefficients of the
        weight vector have not been masked out, i.e. the support of
        ``self.coef_``.

    history_ : list
        History of ``coef_``. ``history_[k]`` contains the values of
        ``coef_`` at iteration k of FROLS.

    Examples
    --------
    >>> import numpy as np
    >>> from scipy.integrate import odeint
    >>> from pysindy import SINDy
    >>> from pysindy.optimizers import FROLS
    >>> lorenz = lambda z,t : [10*(z[1] - z[0]),
    >>>                        z[0]*(28 - z[2]) - z[1],
    >>>                        z[0]*z[1] - 8/3*z[2]]
    >>> t = np.arange(0,2,.002)
    >>> x = odeint(lorenz, [-8,8,27], t)
    >>> opt = FROLS(threshold=.1, alpha=.5)
    >>> model = SINDy(optimizer=opt)
    >>> model.fit(x, t=t[1]-t[0])
    >>> model.print()
    x0' = -9.999 1 + 9.999 x0
    x1' = 27.984 1 + -0.996 x0 + -1.000 1 x1
    x2' = -2.666 x1 + 1.000 1 x0
    """

    def __init__(
        self,
        normalize=False,
        normalize_columns=False,
        fit_intercept=False,
        copy_X=True,
        L0_penalty=None,
        max_iter=10,
    ):
        super(FROLS, self).__init__(
            normalize=normalize,
            fit_intercept=fit_intercept,
            copy_X=copy_X,
            max_iter=max_iter,
        )
        self.normalize_columns = normalize_columns
        self.L0_penalty = L0_penalty
        if self.max_iter <= 0:
            raise ValueError("Max iteration must be > 0")

    def _normed_cov(self, a, b):
        return np.vdot(a, b) / np.vdot(a, a)

    def _select_function(self, x, y, sigma, skip=[]):
        n_features = x.shape[1]
        g = np.zeros(n_features)  # Coefficients to orthogonalized functions
        error = np.zeros(n_features)  # Error reduction ratio at this step
        for m in range(n_features):
            if m not in skip:
                g[m] = self._normed_cov(x[:, m], y)
                error[m] = (
                    abs(g[m]) ** 2 * np.real(np.vdot(x[:, m], x[:, m])) / sigma
                )  # Error reduction

        L = np.argmax(error)  # Select best function

        # Return index of best function, along with ERR and coefficient
        return L, error[L], g[L]

    def _orthogonalize(self, vec, Q):
        """
        Orthogonalize vec with respect to columns of Q
        """
        Qs = vec.copy()
        s = Q.shape[1]
        for r in range(s):
            Qs -= self._normed_cov(Q[:, r], Qs) * Q[:, r]
        return Qs

    def _reduce(self, x, y):
        """Performs at most n_feature iterations of the
        greedy Forward Regression Orthogonal Least Squares (FROLS) algorithm
        """
        n_samples, n_features = x.shape
        n_targets = y.shape[1]
        self.Theta = x
        x_normed = np.copy(x)
        if self.normalize_columns:
            reg = np.zeros(n_features)
            for i in range(n_features):
                reg[i] = 1.0 / np.linalg.norm(x[:, i], 2)
                x_normed[:, i] = reg[i] * x[:, i]

        self.history_ = np.zeros((n_features, n_targets, n_features))
        coef = np.zeros_like(self.coef_)
        for k in range(n_targets):
            # Initialize arrays
            err_glob = np.zeros(n_features)
            g_glob = np.zeros(
                n_features
            )  # Coefficients for selected (orthogonal) functions
            L = np.zeros(
                n_features, dtype=int
            )  # Order of selection, i.e. l[0] is the first, l[1] second...
            A = np.zeros(
                (n_features, n_features)
            )  # Used for inversion to original function set (A @ coef = g)

            # Orthogonal function libraries
            Q = np.zeros_like(x)  # Global library (built over time)
            Qs = np.zeros_like(x)  # Same, but for each step
            sigma = np.real(np.dot(y[:, k], y[:, k]))  # Variance of the signal
            for i in range(n_features):
                for m in range(n_features):
                    if m not in L[:i]:
                        # Orthogonalize with respect to already selected functions
                        Qs[:, m] = self._orthogonalize(x_normed[:, m], Q[:, :i])

                L[i], err_glob[i], g_glob[i] = self._select_function(
                    Qs, y[:, k], sigma, L[:i]
                )

                # Store transformation from original functions to orthogonal library
                for j in range(i):
                    A[j, i] = self._normed_cov(Q[:, j], x_normed[:, L[i]])
                A[i, i] = 1.0
                Q[:, i] = Qs[:, L[i]].copy()
                Qs *= 0.0

                # Invert orthogonal coefficient vector
                # to get coefficients for original functions
                alpha = np.linalg.lstsq(A[:i, :i], g_glob[:i], rcond=1e-6)[0]

                coef_k = np.zeros_like(g_glob)
                coef_k[L[:i]] = alpha
                coef_k[abs(coef_k) < 1e-10] = 0

                # Indicator of selected terms
                ind = np.zeros(n_features, dtype=int)
                ind[L[:i]] = 1

                if self.normalize_columns:
                    self.history_[i, k, :] = np.multiply(reg, np.copy(coef_k))
                else:
                    self.history_[i, k, :] = np.copy(coef_k)

                if i >= self.max_iter:
                    break

        # Figure out lowest MSE coefficients
        err = np.zeros(n_features)
        if self.L0_penalty is not None:
            l0_penalty = self.L0_penalty * np.linalg.cond(x)
        else:
            l0_penalty = 0.0
        for i in range(n_features):
            coef_i = np.asarray(self.history_[i, :, :])
            err[i] = np.sum((y - x @ coef_i.T) ** 2) + l0_penalty * np.count_nonzero(
                coef_i
            )
        self.err_history_ = err
        err_min = np.argmin(err)
        coef = np.asarray(self.history_[err_min, :, :])
        if self.normalize_columns:
            self.coef_ = np.multiply(reg, coef)
        else:
            self.coef_ = coef
