import warnings
from itertools import combinations as combo_nr
from itertools import product
from itertools import repeat
from math import comb
from typing import cast
from typing import NewType
from typing import Optional
from typing import Union

import cvxpy as cp
import numpy as np
from numpy.typing import NBitBase
from numpy.typing import NDArray
from sklearn.exceptions import ConvergenceWarning

from ..feature_library.polynomial_library import n_poly_features
from ..feature_library.polynomial_library import PolynomialLibrary
from ..utils import reorder_constraints
from .constrained_sr3 import ConstrainedSR3

AnyFloat = np.dtype[np.floating[NBitBase]]
Int1D = np.ndarray[tuple[int], np.dtype[np.int_]]
Float1D = np.ndarray[tuple[int], AnyFloat]
Float2D = np.ndarray[tuple[int, int], AnyFloat]
Float3D = np.ndarray[tuple[int, int, int], AnyFloat]
Float4D = np.ndarray[tuple[int, int, int, int], AnyFloat]
Float5D = np.ndarray[tuple[int, int, int, int, int], AnyFloat]
FloatND = NDArray[np.floating[NBitBase]]
NFeat = NewType("NFeat", int)
NTarget = NewType("NTarget", int)


class TrappingSR3(ConstrainedSR3):
    """
    Generalized trapping variant of sparse relaxed regularized regression.
    This optimizer can be used to identify quadratically nonlinear systems with
    either a-priori globally or locally stable (bounded) solutions.

    This optimizer can be used to minimize five different objective functions:

    .. math::

        0.5\\|y-Xw\\|^2_2 + \\lambda \\times R(w)
        + 0.5\\|Pw-A\\|^2_2/\\eta + \\delta_0(Cw-d)
        + \\delta_{\\Lambda}(A) + \\alpha \\|Qijk\\|
        + \\beta \\|Q_{ijk} + Q_{jik} + Q_{kij}\\|

    where :math:`R(w)` is a regularization function, C is a constraint matrix
    detailing affine constraints on the model coefficients, A is a proxy for
    the quadratic contributions to the energy evolution, and
    :math:`Q_{ijk}` are the quadratic coefficients in the model. For
    provably globally bounded solutions, use :math:`\\alpha >> 1`,
    :math:`\\beta >> 1`, and equality constraints. For maximizing the local
    stability radius of the model one has the choice to do this by
    (1) minimizing the values in :math:`Q_{ijk}`, (2) promoting models
    with skew-symmetrix :math:`Q_{ijk}` coefficients, or
    (3) using inequality constraints for skew-symmetry in :math:`Q_{ijk}`.

    See the following references for more details:

        Kaptanoglu, Alan A., et al. "Promoting global stability in
        data-driven models of quadratic nonlinear dynamics."
        arXiv preprint arXiv:2105.01843 (2021).

    Parameters
    ----------
    eta :
        Determines the strength of the stability term :math:`||Pw-A||^2` in the
        optimization. The default value is very large so that the
        algorithm default is to ignore the stability term. In this limit,
        this should be approximately equivalent to the ConstrainedSR3 method.

    eps_solver :
        If threshold != 0, this specifies the error tolerance in the
        CVXPY (OSQP) solve. Default 1.0e-7

    alpha:
        Determines the strength of the local stability term :math:`||Qijk||^2`
        in the optimization. The default value (1e20) is very large so that the
        algorithm default is to ignore this term.

    beta:
        Determines the strength of the local stability term
        :math:`||Qijk + Qjik + Qkij||^2` in the
        optimization. The default value is very large so that the
        algorithm default is to ignore this term.

    mod_matrix:
        Lyapunov matrix.  Trapping theorems apply to energy
        :math:`\\propto \\dot y \\cdot y`, but also to any
        :math:`\\propto \\dot y P \\cdot y` for Lyapunov matrix :math:`P`.
        Defaults to the identity matrix.

    alpha_A :
        Determines the step size in the prox-gradient descent over A.
        For convergence, need alpha_A <= eta, so default
        alpha_A = eta is used.

    alpha_m :
        Determines the step size in the prox-gradient descent over m.
        For convergence, need alpha_m <= eta / ||w^T * PQ^T * PQ * w||.
        Typically 0.01 * eta <= alpha_m <= 0.1 * eta.  (default eta * 0.1)

    gamma :
        Determines the negative interval that matrix A is projected onto.
        For most applications gamma = 0.1 - 1.0 works pretty well.

    tol_m :
        Tolerance used for determining convergence of the optimization
        algorithm over m.

    thresholder :
        Regularization function to use. For current trapping SINDy,
        only the L1 and L2 norms are implemented. Note that other convex norms
        could be straightforwardly implemented, but L0 requires
        reformulation because of nonconvexity. (default 'L1')

    m0 :
        Initial guess for trap center in the optimization. Default None
        initializes vector elements randomly in [-1, 1]. shape (n_targets)

    A0 :
        Initial guess for vector A in the optimization.  Shape (n_targets, n_targets)
        Default None, meaning A is initialized as A = diag(gamma).

    Attributes
    ----------
    A_history_ : list
        History of the auxiliary variable A that approximates diag(PW).

    m_history_ : list
        History of the shift vector m that determines the origin of the
        trapping region.

    PW_history_ : list
        History of PW = A^S, the quantity we are attempting to make
        negative definite.

    PWeigs_history_ : list
        History of diag(PW), a list of the eigenvalues of A^S at
        each iteration. Tracking this allows us to ascertain if
        A^S is indeed being pulled towards the space of negative
        definite matrices.

    PL_unsym_ : np.ndarray, shape (n_targets, n_targets, n_targets, n_features)
        Unsymmetrized linear coefficient part of the P matrix in :math:``||Pw - A||^2``

    PL_ : np.ndarray, shape (n_targets, n_targets, n_targets, n_features)
        Linear coefficient part of the P matrix in :math:``||Pw - A||^2``

    PQ_ : np.ndarray, shape (n_targets, n_targets,
                            n_targets, n_targets, n_features)
        Quadratic coefficient part of the P matrix in :math:``||Pw - A||^2``

    PT_ : np.ndarray, shape (n_targets, n_targets,
                            n_targets, n_targets, n_features)
        Transpose of 1st dimension and 2nd dimension of quadratic coefficient
        part of the P matrix in :math:``||Pw - A||^2``

    objective_history_ : list
        History of the value of the objective at each step. Note that
        the trapping SINDy problem is nonconvex, meaning that this value
        may increase and decrease as the algorithm works.

    Examples
    --------
    >>> import numpy as np
    >>> from scipy.integrate import odeint
    >>> from pysindy import SINDy
    >>> from pysindy.optimizers import TrappingSR3
    >>> lorenz = lambda z,t : [10*(z[1] - z[0]),
    >>>                        z[0]*(28 - z[2]) - z[1],
    >>>                        z[0]*z[1] - 8/3*z[2]]
    >>> t = np.arange(0,2,.002)
    >>> x = odeint(lorenz, [-8,8,27], t)
    >>> opt = TrappingSR3(threshold=0.1)
    >>> model = SINDy(optimizer=opt)
    >>> model.fit(x, t=t[1]-t[0])
    >>> model.print()
    x0' = -10.004 1 + 10.004 x0
    x1' = 27.994 1 + -0.993 x0 + -1.000 1 x1
    x2' = -2.662 x1 + 1.000 1 x0
    """

    def __init__(
        self,
        *,
        _n_tgts: int = None,
        _include_bias: bool = False,
        _interaction_only: bool = False,
        method: str = "global",
        eta: Union[float, None] = None,
        eps_solver: float = 1e-7,
        alpha: Optional[float] = None,
        beta: Optional[float] = None,
        mod_matrix: Optional[NDArray] = None,
        alpha_A: Union[float, None] = None,
        alpha_m: Union[float, None] = None,
        gamma: float = -0.1,
        tol_m: float = 1e-5,
        thresholder: str = "l1",
        m0: Union[NDArray, None] = None,
        A0: Union[NDArray, None] = None,
        **kwargs,
    ):
        # n_tgts, constraints, etc are data-dependent parameters and belong in
        # _reduce/fit ().  The following is a hack until we refactor how
        # constraints are applied in ConstrainedSR3 and MIOSR
        self._include_bias = _include_bias
        self._interaction_only = _interaction_only
        self._n_tgts = _n_tgts
        self.mod_matrix = mod_matrix
        if _n_tgts is None:
            warnings.warn(
                "Trapping Optimizer initialized without _n_tgts.  It will likely"
                " be unable to fit data"
            )
            self._n_tgts = 1
        if self.mod_matrix is None:
            self.mod_matrix = np.eye(self._n_tgts)
        lsv, sing_vals, rsv = np.linalg.svd(self.mod_matrix)
        # scipy.linalg.sqrtm
        self.rt_mod_mat = lsv @ np.diag(np.sqrt(sing_vals)) @ rsv
        self.rt_inv_mod_mat = lsv @ np.diag(np.sqrt(1 / sing_vals)) @ rsv

        if method == "global":
            if hasattr(kwargs, "constraint_separation_index"):
                constraint_separation_index = kwargs["constraint_separation_index"]
            elif kwargs.get("inequality_constraints", False):
                constraint_separation_index = kwargs["constraint_lhs"].shape[0]
            else:
                constraint_separation_index = 0
            constraint_rhs, constraint_lhs = _make_constraints(
                self._n_tgts, include_bias=_include_bias
            )
            constraint_lhs = np.tensordot(constraint_lhs, self.mod_matrix, axes=1)
            constraint_order = kwargs.pop("constraint_order", "feature")
            if constraint_order == "target":
                constraint_lhs = np.transpose(constraint_lhs, [0, 2, 1])
            constraint_lhs = np.reshape(constraint_lhs, (constraint_lhs.shape[0], -1))
            try:
                constraint_lhs = np.concatenate(
                    (kwargs.pop("constraint_lhs"), constraint_lhs), 0
                )
                constraint_rhs = np.concatenate(
                    (kwargs.pop("constraint_rhs"), constraint_rhs), 0
                )
            except KeyError:
                pass

            super().__init__(
                constraint_lhs=constraint_lhs,
                constraint_rhs=constraint_rhs,
                constraint_separation_index=constraint_separation_index,
                constraint_order=constraint_order,
                equality_constraints=True,
                thresholder=thresholder,
                **kwargs,
            )
            self.method = "global"
        elif method == "local":
            super().__init__(thresholder=thresholder, **kwargs)
            self.method = "local"
        else:
            raise ValueError(f"Can either use 'global' or 'local' method, not {method}")

        self.eps_solver = eps_solver
        self.m0 = m0
        self.A0 = A0
        self.alpha_A = alpha_A
        self.alpha_m = alpha_m
        self.eta = eta
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.tol_m = tol_m
        self.__post_init_guard()

    def __post_init_guard(self):
        """Conduct initialization post-init, as required by scikitlearn API"""
        if self.thresholder.lower() not in ("l1", "l2", "weighted_l1", "weighted_l2"):
            raise ValueError("Regularizer must be (weighted) L1 or L2")
        if self.eta is None:
            warnings.warn(
                "eta was not set, so defaulting to eta = 1e20 "
                "with alpha_m = 1e-2 * eta, alpha_A = eta. Here eta is so "
                "large that the stability term in the optimization "
                "will be ignored."
            )
            self.eta = 1e20
            self.alpha_m = 1e18
            self.alpha_A = 1e20
        else:
            if self.alpha_m is None:
                self.alpha_m = self.eta * 1e-2
            if self.alpha_A is None:
                self.alpha_A = self.eta
        if self.eta <= 0:
            raise ValueError("eta must be positive")
        if self.alpha is None:
            self.alpha = 1e20
            warnings.warn(
                "alpha was not set, so defaulting to alpha = 1e20 "
                "which is so"
                "large that the ||Qijk|| term in the optimization "
                "will be essentially ignored."
            )
        if self.beta is None:
            self.beta = 1e20
            warnings.warn(
                "beta was not set, so defaulting to beta = 1e20 "
                "which is so"
                "large that the ||Qijk + Qjik + Qkij|| "
                "term in the optimization will be essentially ignored."
            )

        if self.alpha_m < 0 or self.alpha_m > self.eta:
            raise ValueError("0 <= alpha_m <= eta")
        if self.alpha_A < 0 or self.alpha_A > self.eta:
            raise ValueError("0 <= alpha_A <= eta")
        if self.gamma >= 0:
            raise ValueError("gamma must be negative")
        if self.tol <= 0 or self.tol_m <= 0 or self.eps_solver <= 0:
            raise ValueError("tol and tol_m must be positive")
        if self.inequality_constraints and self.threshold == 0.0:
            raise ValueError("Inequality constraints requires threshold!=0")
        if self.A0 is None:
            self.A0 = np.diag(self.gamma * np.ones(self._n_tgts))
        if self.m0 is None:
            np.random.seed(1)
            self.m0 = (np.random.rand(self._n_tgts) - np.ones(self._n_tgts)) * 2

    def set_params(self, **kwargs):
        super().set_params(**kwargs)
        self.__post_init_guard()

    @staticmethod
    def _build_PC(polyterms: list[tuple[int, Int1D]]) -> Float3D:
        r"""Build the matrix that projects out the constant term of a library

        Coefficients in each polynomial equation :math:`i\in \{1,\dots, r\}` can
        be stored in an array arranged as written out on paper (e.g.
        :math:` f_i(x) = a^i_0 + a^i_1 x_1, a^i_2 x_1x_2, \dots a^i_N x_r^2`) or
        in a series of matrices :math:`E \in \mathbb R^r`,
        :math:`L\in \mathbb R^{r\times r}`, and (without loss of generality) in
        :math:`Q\in \mathbb R^{r \times r \times r}, where each
        :math:`Q^{(i)}_{j,k}` is symmetric in the last two indexes.

        This function builds the projection tensor for extracting the constant
        terms :math:`E` from a set of coefficients in the first representation.

        Args:
            polyterms: the ordering and meaning of terms in the equations.  Each
                entry represents a term in the equation and comprises its index
                and an array of exponents for each variable

        Returns:
            3rd order tensor
        """
        n_targets, n_features, _, _, _ = _build_lib_info(polyterms)
        c_terms = [ind for ind, exps in polyterms if sum(exps) == 0]
        PC = np.zeros((n_targets, n_targets, n_features))
        if c_terms:  # either a length 0 or length 1 list
            PC[range(n_targets), range(n_targets), c_terms[0]] = 1.0
        return PC

    @staticmethod
    def _build_PL(polyterms: list[tuple[int, Int1D]]) -> tuple[Float4D, Float4D]:
        r"""Build the matrix that projects out the linear terms of a library

        Coefficients in each polynomial equation :math:`i\in \{1,\dots, r\}` can
        be stored in an array arranged as written out on paper (e.g.
        :math:` f_i(x) = a^i_0 + a^i_1 x_1, a^i_2 x_1x_2, \dots a^i_N x_r^2`) or
        in a series of matrices :math:`E \in \mathbb R^r`,
        :math:`L\in \mathbb R^{r\times r}`, and (without loss of generality) in
        :math:`Q\in \mathbb R^{r \times r \times r}, where each
        :math:`Q^{(i)}_{j,k}` is symmetric in the last two indexes.

        This function builds the projection tensor for extracting the linear
        terms :math:`L` from a set of coefficients in the first representation.
        The function also calculates the projection tensor for extracting the
        symmetrized version of L

        Args:
            polyterms: the ordering and meaning of terms in the equations.  Each
                entry represents a term in the equation and comprises its index
                and an array of exponents for each variable

        Returns:
            Two 4th order tensors, the first one symmetric in the first two
            indexes.
        """
        n_targets, n_features, lin_terms, _, _ = _build_lib_info(polyterms)
        PL_tensor_unsym = np.zeros((n_targets, n_targets, n_targets, n_features))
        tgts = range(n_targets)
        for j in range(n_targets):
            PL_tensor_unsym[tgts, j, tgts, lin_terms[j]] = 1
        PL_tensor = (PL_tensor_unsym + np.transpose(PL_tensor_unsym, [1, 0, 2, 3])) / 2
        return cast(Float4D, PL_tensor), cast(Float4D, PL_tensor_unsym)

    @staticmethod
    def _build_PQ(polyterms: list[tuple[int, Int1D]]) -> Float5D:
        r"""Build the matrix that projects out the quadratic terms of a library

        Coefficients in each polynomial equation :math:`i\in \{1,\dots, r\}` can
        be stored in an array arranged as written out on paper (e.g.
        :math:` f_i(x) = a^i_0 + a^i_1 x_1, a^i_2 x_1x_2, \dots a^i_N x_r^2`) or
        in a series of matrices :math:`E \in \mathbb R^r`,
        :math:`L\in \mathbb R^{r\times r}`, and (without loss of generality) in
        :math:`Q\in \mathbb R^{r \times r \times r}, where each
        :math:`Q^{(i)}_{j,k}` is symmetric in the last two indexes.

        This function builds the projection tensor for extracting the quadratic
        forms :math:`Q` from a set of coefficients in the first representation.

        Args:
            polyterms: the ordering and meaning of terms in the equations.  Each
                entry represents a term in the equation and comprises its index
                and an array of exponents for each variable

        Returns:
            5th order tensor symmetric in second and third indexes.
        """
        n_targets, n_features, _, pure_terms, mixed_terms = _build_lib_info(polyterms)
        PQ = np.zeros((n_targets, n_targets, n_targets, n_targets, n_features))
        tgts = range(n_targets)
        for j, k in product(*repeat(range(n_targets), 2)):
            if j == k:
                PQ[tgts, j, k, tgts, pure_terms[j]] = 1.0
            if j != k:
                PQ[tgts, j, k, tgts, mixed_terms[frozenset({j, k})]] = 1 / 2
        return cast(Float5D, PQ)

    def _set_Ptensors(
        self, n_targets: int
    ) -> tuple[Float3D, Float4D, Float4D, Float5D, Float5D, Float5D]:
        """Make the projection tensors used for the algorithm."""
        lib = PolynomialLibrary(2, include_bias=self._include_bias).fit(
            np.zeros((1, n_targets))
        )
        polyterms = [(t_ind, exps) for t_ind, exps in enumerate(lib.powers_)]

        PC_tensor = self._build_PC(polyterms)
        PL_tensor, PL_tensor_unsym = self._build_PL(polyterms)
        PQ_tensor = self._build_PQ(polyterms)
        PT_tensor = PQ_tensor.transpose([1, 0, 2, 3, 4])
        # PM is the sum of PQ and PQ which projects out the sum of Qijk and Qjik
        PM_tensor = cast(Float5D, PQ_tensor + PT_tensor)

        return PC_tensor, PL_tensor_unsym, PL_tensor, PQ_tensor, PT_tensor, PM_tensor

    def _update_A(self, A_old, PW):
        """Update the symmetrized A matrix"""
        eigvals, eigvecs = np.linalg.eigh(A_old)
        eigPW, eigvecsPW = np.linalg.eigh(PW)
        r = A_old.shape[0]
        A = np.diag(eigvals)
        for i in range(r):
            if eigvals[i] > self.gamma:
                A[i, i] = self.gamma
        return eigvecsPW @ A @ np.linalg.inv(eigvecsPW)

    def _convergence_criterion(self):
        """Calculate the convergence criterion for the optimization over w"""
        this_coef = self.history_[-1]
        if len(self.history_) > 1:
            last_coef = self.history_[-2]
        else:
            last_coef = np.zeros_like(this_coef)
        err_coef = np.sqrt(np.sum((this_coef - last_coef) ** 2))
        return err_coef

    def _m_convergence_criterion(self):
        """Calculate the convergence criterion for the optimization over m"""
        return np.sum(np.abs(self.m_history_[-2] - self.m_history_[-1]))

    def _objective(self, x, y, coef_sparse, A, PW, k):
        """Objective function"""
        # Compute the errors
        sindy_loss = (y - np.dot(x, coef_sparse)) ** 2
        relax_loss = (A - PW) ** 2
        Qijk = np.einsum("ya,abcde,ed", self.mod_matrix, self.PQ_, coef_sparse)
        Qijk_permsum = (
            Qijk + np.transpose(Qijk, [1, 2, 0]) + np.transpose(Qijk, [2, 0, 1])
        ) ** 2
        L1 = self.threshold * np.sum(np.abs(coef_sparse.flatten()))
        sindy_loss = 0.5 * np.sum(sindy_loss)
        relax_loss = 0.5 * np.sum(relax_loss) / self.eta
        nonlin_ens_loss = 0.5 * np.sum(Qijk**2) / self.alpha
        cubic_ens_loss = 0.5 * np.sum(Qijk_permsum) / self.beta

        obj = sindy_loss + relax_loss + L1
        if self.method == "local":
            obj += nonlin_ens_loss + nonlin_ens_loss

        if self.verbose and k % max(1, self.max_iter // 10) == 0:
            print(
                f"{k:5d} ... {sindy_loss:8.3e} ... {relax_loss:8.3e} ... {L1:8.2e}"
                f" ... {nonlin_ens_loss:8.2e} ... {cubic_ens_loss:8.2e} ... {obj:8.2e}"
            )
        return obj

    def _update_coef_sparse_rs(
        self, n_tgts, n_features, var_len, x_expanded, y, Pmatrix, A, coef_prev
    ):
        """Solve coefficient update with CVXPY if threshold != 0"""
        xi, cost = self._create_var_and_part_cost(var_len, x_expanded, y)
        cost = cost + cp.sum_squares(Pmatrix @ xi - A.flatten()) / self.eta

        # new terms minimizing quadratic piece ||P^Q @ xi||_2^2
        if self.method == "local":
            Q = np.reshape(
                self.PQ_, (n_tgts * n_tgts * n_tgts, n_features * n_tgts), "F"
            )
            cost = cost + cp.sum_squares(Q @ xi) / self.alpha
            Q = np.reshape(self.PQ_, (n_tgts, n_tgts, n_tgts, n_features * n_tgts), "F")
            Q_ep = Q + np.transpose(Q, [1, 2, 0, 3]) + np.transpose(Q, [2, 0, 1, 3])
            Q_ep = np.reshape(
                Q_ep, (n_tgts * n_tgts * n_tgts, n_features * n_tgts), "F"
            )
            cost = cost + cp.sum_squares(Q_ep @ xi) / self.beta

        return self._update_coef_cvxpy(xi, cost, var_len, coef_prev, self.eps_solver)

    def _update_coef_nonsparse_rs(
        self,
        x: Float2D,
        y: Float2D,
        P_A: Float4D,
        quad_energy_coeff_A: Float2D,
    ):
        """Solve a partial minimization for w, the SINDy coefficients

        Letting :math:`Q_{ijk} =P_Q w` and :math:`A=P_A w`, Partially minimizes the sum
        of:
        * Error in predicting SINDy dynamics, :math:`||Xw-y||`
        * Deviation from previous iterate of quadratic energy coefficients,
            :math:`||A-A^{-}||^2`
        * (Optionally) the size of the quadratic ODE terms, :math:`||Q_{ijk}||^2`
        * (Optionally) the symmetry-breaking of the quadratic ODE terms,
            :math:`||Q_{ijk} + Q_{jki} + Q_{kij}||^2`

        Args:
            x: The samples of SINDy features
            y: The SINDy derivatives
            P_A: Projects coefficients w onto energy quadratic form.
            quad_energy_coeff_A: The energy quadratic form from the previous
                iteration of trapping SINDy
            coef_prev

        """
        _, _, n_tgts, n_features = P_A.shape
        var_len = n_tgts * n_features

        # Input variable still has 2 dimensions, so hessians have 4
        hess = np.zeros((n_tgts, n_features, n_tgts, n_features))
        xTx = x.T @ x
        for tgt in range(n_tgts):
            hess[tgt, :, tgt, :] = xTx
        pTp = np.einsum("abcd,baef->cdef", P_A, P_A)
        hess += pTp / self.eta
        if self.method == "local":
            PQTPQ = np.tensordot(self.PQ_, self.PQ_, axes=([0, 1, 2], [0, 1, 2]))
            PQ = np.einsum("ya,abcde->ybcde", self.mod_matrix, self.PQ_)
            PQ_ep = (
                PQ
                + np.transpose(PQ, [1, 2, 0, 3, 4])
                + np.transpose(PQ, [2, 0, 1, 3, 4])
            )
            PQTPQ_ep = np.tensordot(PQ_ep, PQ_ep, axes=([0, 1, 2], [0, 1, 2]))
            hess += PQTPQ / self.alpha + PQTPQ_ep / self.beta

        PaTA = np.einsum("bacd,ab->cd", P_A, quad_energy_coeff_A)
        # We are still, across most of SR3, handling (NFeat,NTarget) shaped arrays
        xTy = x.T @ y
        PaTA = np.transpose(PaTA)
        grad_const = xTy + PaTA / self.eta
        hess = np.transpose(hess, [1, 0, 3, 2])
        hess = np.reshape(hess, (var_len, var_len))
        grad_const = np.reshape(grad_const, (var_len))
        coef_flat = self._solve_nonsparse_relax_and_split(hess, grad_const)
        return coef_flat.reshape(n_features, n_tgts)

    def _solve_m_relax_and_split(
        self,
        trap_ctr: Float1D,
        A: Float2D,
        coef_sparse: np.ndarray[tuple[NFeat, NTarget], AnyFloat],
    ) -> tuple[Float1D, Float2D]:
        """Solves the (m, A) algorithm update.

        TODO: explain the optimization this solves, add good names to variables,
        and refactor/indirect the if global/local trapping conditionals

        Returns the new trap center (m) and the new A
        """
        # prox-gradient descent for (A, m)
        # Calculate projection matrix from Quad terms to As
        mPM = np.tensordot(self.PM_, trap_ctr, axes=([2], [0]))
        p = self.PL_ + mPM
        p = np.einsum("ya,abcd,bz->yzcd", self.rt_mod_mat, p, self.rt_inv_mod_mat)
        p = (p + np.transpose(p, [1, 0, 2, 3])) / 2
        PW = np.tensordot(p, coef_sparse, axes=([3, 2], [0, 1]))

        # Calculate As and its quad term components
        PMW = np.tensordot(self.PM_, coef_sparse, axes=([4, 3], [0, 1]))
        PMW = np.einsum("ya,abc,bz->yzc", self.rt_mod_mat, PMW, self.rt_inv_mod_mat)
        PMW = (PMW + np.transpose(PMW, [1, 0, 2])) / 2
        # Calculate error in quadratic balance, and adjust trap center
        A_b = (A - PW) / self.eta
        # PQWT_PW is gradient of some loss in m
        PMT_PW = np.tensordot(PMW, A_b, axes=([2, 1], [0, 1]))
        trap_new = trap_ctr - self.alpha_m * PMT_PW

        # Update A
        A_new = self._update_A(A - self.alpha_A * A_b, PW)
        return trap_new, A_new

    def _solve_nonsparse_relax_and_split(self, hess, gradient_constant):
        """Update for the coefficients if threshold = 0."""
        if self.use_constraints:
            coef_nonsparse = _equality_constrained_linlsq(
                hess, gradient_constant, self.constraint_lhs, self.constraint_rhs
            )
        else:
            hess_inv = np.linalg.pinv(hess, rcond=1e-10)
            coef_nonsparse = hess_inv.dot(gradient_constant)
        return coef_nonsparse

    def _reduce(self, x, y):
        """
        Perform at most ``self.max_iter`` iterations of the
        TrappingSR3 algorithm.
        Assumes initial guess for coefficients is stored in ``self.coef_``.
        """
        self.A_history_ = []
        self.m_history_ = []
        self.p_history_ = []
        self.PW_history_ = []
        self.PWeigs_history_ = []
        self.history_ = []
        n_samples, n_tgts = y.shape
        n_features = n_poly_features(
            n_tgts,
            2,
            include_bias=self._include_bias,
            interaction_only=self._interaction_only,
        )
        var_len = n_features * n_tgts

        (
            self.PC_,
            self.PL_unsym_,
            self.PL_,
            self.PQ_,
            self.PT_,
            self.PM_,
        ) = self._set_Ptensors(n_tgts)

        # Set initial coefficients
        if self.use_constraints and self.constraint_order.lower() == "target":
            self.constraint_lhs = reorder_constraints(
                self.constraint_lhs, n_features, output_order="feature"
            )
        coef_sparse: np.ndarray[tuple[NFeat, NTarget], AnyFloat] = self.coef_.T

        # Print initial values for each term in the optimization
        if self.verbose:
            row = [
                "Iter",
                "|y-Xw|^2",
                "|Pw-A|^2/eta",
                "|w|_1",
                "|Qijk|/a",
                "|Qijk+...|/b",
                "Total:",
            ]
            print(
                "{: >5} ... {: >8} ... {: >10} ... {: >5}"
                " ... {: >8} ... {: >10} ... {: >8}".format(*row)
            )

        A = self.A0
        self.A_history_.append(A)
        trap_ctr = self.m0
        self.m_history_.append(trap_ctr)

        # Precompute some objects for optimization
        x_expanded = np.zeros((n_samples, n_tgts, n_features, n_tgts))
        for i in range(n_tgts):
            x_expanded[:, i, :, i] = x
        x_expanded = np.reshape(x_expanded, (n_samples * n_tgts, n_tgts * n_features))

        # keep track of last solution in case method fails
        trap_prev_ctr = trap_ctr

        # Begin optimization loop
        objective_history = []
        for k in range(self.max_iter):
            # update P tensor from the newest trap center
            mPM = np.tensordot(self.PM_, trap_ctr, axes=([2], [0]))
            P_A = np.einsum(
                "ya,abcd,bz->yzcd", self.rt_mod_mat, self.PL_ + mPM, self.rt_inv_mod_mat
            )
            P_A = (P_A + np.transpose(P_A, [1, 0, 2, 3])) / 2
            Pmatrix = P_A.reshape(n_tgts * n_tgts, n_tgts * n_features)
            self.p_history_.append(P_A)

            coef_prev = coef_sparse
            if (self.threshold > 0.0) or self.inequality_constraints:
                coef_sparse = self._update_coef_sparse_rs(
                    n_tgts, n_features, var_len, x_expanded, y, Pmatrix, A, coef_prev
                )
            else:
                coef_sparse = self._update_coef_nonsparse_rs(x, y, P_A, A)

            # If problem over xi becomes infeasible, break out of the loop
            if coef_sparse is None:
                coef_sparse = coef_prev
                break

            trap_ctr, A = self._solve_m_relax_and_split(trap_ctr, A, coef_sparse)

            # If problem over m becomes infeasible, break out of the loop
            if trap_ctr is None:
                trap_ctr = trap_prev_ctr
                break
            self.history_.append(coef_sparse.T)
            PW = np.tensordot(P_A, coef_sparse, axes=([3, 2], [0, 1]))

            # (m,A) update finished, append the result
            self.m_history_.append(trap_ctr)
            self.A_history_.append(A)
            eigvals, eigvecs = np.linalg.eig(PW)
            self.PW_history_.append(PW)
            self.PWeigs_history_.append(np.sort(eigvals))

            # update objective
            objective_history.append(self._objective(x, y, coef_sparse, A, PW, k))

            if (
                self._m_convergence_criterion() < self.tol_m
                and self._convergence_criterion() < self.tol
            ):
                break
        else:
            warnings.warn(
                f"TrappingSR3 did not converge after {self.max_iter} iters.",
                ConvergenceWarning,
            )

        self.coef_ = coef_sparse.T
        self.objective_history = objective_history


def _make_constraints(n_tgts: int, **kwargs):
    """Create constraints for the Quadratic terms in TrappingSR3.

    These are the constraints from equation 5 of the Trapping SINDy paper.

    Args:
        n_tgts: number of coordinates or modes for which you're fitting an ODE.
        kwargs: Keyword arguments to PolynomialLibrary such as
            ``include_bias``.

    Returns:
        A tuple of the constraint zeros, and a constraint matrix to multiply
        by the coefficient matrix of Polynomial terms. Number of constraints is
        ``n_tgts + 2 * math.comb(n_tgts, 2) + math.comb(n_tgts, 3)``.
        Constraint matrix is of shape ``(n_constraint, n_feature, n_tgt)``.
        To get "feature" order constraints, use
        ``np.reshape(constraint_matrix, (n_constraints, -1))``.
        To get "target" order constraints, transpose axis 1 and 2 before
        reshaping.
    """
    lib = PolynomialLibrary(2, **kwargs).fit(np.zeros((1, n_tgts)))
    terms = [(t_ind, exps) for t_ind, exps in enumerate(lib.powers_)]
    _, n_terms, linear_terms, pure_terms, mixed_terms = _build_lib_info(terms)
    # index of tgt -> index of its pure quadratic term
    pure_terms = {np.argmax(exps): t_ind for t_ind, exps in terms if max(exps) == 2}
    # two indexes of tgts -> index of their mixed quadratic term
    mixed_terms = {
        frozenset(np.argwhere(exponent == 1).flatten()): t_ind
        for t_ind, exponent in terms
        if max(exponent) == 1 and sum(exponent) == 2
    }
    constraint_mat = np.vstack(
        (
            _pure_constraints(n_tgts, n_terms, pure_terms),
            _antisymm_double_constraint(n_tgts, n_terms, pure_terms, mixed_terms),
            _antisymm_triple_constraints(n_tgts, n_terms, mixed_terms),
        )
    )

    return np.zeros(len(constraint_mat)), constraint_mat


def _pure_constraints(n_tgts: int, n_terms: int, pure_terms: dict[int, int]) -> Float2D:
    """Set constraints for coefficients adorning terms like a_i^3 = 0"""
    constraint_mat = np.zeros((n_tgts, n_terms, n_tgts))
    for constr_ind, (tgt_ind, term_ind) in zip(range(n_tgts), pure_terms.items()):
        constraint_mat[constr_ind, term_ind, tgt_ind] = 1.0
    return constraint_mat


def _antisymm_double_constraint(
    n_tgts: int,
    n_terms: int,
    pure_terms: dict[int, int],
    mixed_terms: dict[frozenset[int], int],
) -> Float2D:
    """Set constraints for coefficients adorning terms like a_i^2 * a_j=0"""
    constraint_mat_1 = np.zeros((len(mixed_terms), n_terms, n_tgts))  # a_i^2 * a_j
    constraint_mat_2 = np.zeros((len(mixed_terms), n_terms, n_tgts))  # a_i * a_j^2
    for constr_ind, ((tgt_i, tgt_j), mix_term) in enumerate(mixed_terms.items()):
        constraint_mat_1[constr_ind, mix_term, tgt_i] = 1.0
        constraint_mat_1[constr_ind, pure_terms[tgt_i], tgt_j] = 1.0
        constraint_mat_2[constr_ind, mix_term, tgt_j] = 1.0
        constraint_mat_2[constr_ind, pure_terms[tgt_j], tgt_i] = 1.0

    return np.concatenate((constraint_mat_1, constraint_mat_2), axis=0)


def _antisymm_triple_constraints(
    n_tgts: int, n_terms: int, mixed_terms: dict[frozenset[int], int]
) -> Float2D:
    constraint_mat = np.zeros((comb(n_tgts, 3), n_terms, n_tgts))  # a_ia_ja_k

    def find_symm_term(a, b):
        return mixed_terms[frozenset({a, b})]

    for constr_ind, (tgt_i, tgt_j, tgt_k) in enumerate(combo_nr(range(n_tgts), 3)):
        constraint_mat[constr_ind, find_symm_term(tgt_j, tgt_k), tgt_i] = 1
        constraint_mat[constr_ind, find_symm_term(tgt_k, tgt_i), tgt_j] = 1
        constraint_mat[constr_ind, find_symm_term(tgt_i, tgt_j), tgt_k] = 1

    return constraint_mat


def _build_lib_info(
    polyterms: list[tuple[int, Int1D]]
) -> tuple[int, int, dict[int, int], dict[int, int], dict[frozenset[int], int]]:
    """From polynomial, calculate various useful info

    Args:
        polyterms.  The output of PolynomialLibrary.powers_.  Each term is
            a tuple of it's index in the ordering and a 1D array of the
            exponents of each feature.

    Returns:
        the number of targets
        the number of features
        a dictionary from each target to its linear term index
        a dictionary from each target to its quadratic term index
        a dictionary from each pair of targets to its mixed term index
    """
    try:
        n_targets = len(polyterms[0][1])
    except IndexError:
        raise ValueError("Passed a polynomial library with no terms")
    n_features = len(polyterms)
    mixed_terms = {
        frozenset(np.argwhere(exps == 1).flatten()): t_ind
        for t_ind, exps in polyterms
        if max(exps) == 1 and sum(exps) == 2
    }
    pure_terms = {np.argmax(exps): t_ind for t_ind, exps in polyterms if max(exps) == 2}
    linear_terms = {
        np.argmax(exps): t_ind for t_ind, exps in polyterms if sum(exps) == 1
    }
    return n_targets, n_features, linear_terms, pure_terms, mixed_terms


def _equality_constrained_linlsq(
    hess: Float2D, grad_const: Float2D, constraint_lhs: Float2D, constraint_rhs: Float2D
):
    """Solve the constrained least squares problem via lagrange multipliers

    For the inversion of the lagrange gradient matrix, see
    `wikipedia <https://en.wikipedia.org/wiki/Block_matrix#Inversion>`_

    Arguments:
        hess: the hessian of the loss term.  For regression Ax = b, this is A^TA.
            Must be a square (positive definite) matrix
        grad_const: the constant part of the gradient of the loss term.  For
            regression Ax = b, this is A^Tb.  Must be a column vector.
        constraint_lhs: matrix on left hand side of constraint equation Cx=d.
            Must have same second dimension as hess
        constraint_rhs: vector on right hand side of constraint equation Cx=d.
            Must be a column vector.

    Returns:
        Column vector
    """
    C = constraint_lhs
    d = constraint_rhs
    inv1 = np.linalg.pinv(hess, rcond=1e-10)
    inv2 = np.linalg.pinv(C @ inv1 @ C.T, rcond=1e-10)
    return inv1 @ (grad_const + C.T @ inv2 @ (d - C @ inv1 @ grad_const))
