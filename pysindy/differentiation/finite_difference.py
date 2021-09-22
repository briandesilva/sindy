import numpy as np

from .base import BaseDifferentiation


class FiniteDifference(BaseDifferentiation):
    """Finite difference derivatives.

    For now only first and second order finite difference methods have been
    implemented.

    Parameters
    ----------
    order: int, optional (default 2)
        The order of the finite difference method to be used.
        If 1, first order forward difference will be used.
        If 2, second order centered difference will be used.

    d : int, 1, 2, 3, 4 optional (default 1)
        The order of derivative to take (d > 3 inaccurate).

    is_uniform : boolean, optional (default False)
        Parameter to tell the differentiation that, although a 1D or 2D
        grid is passed, it is uniform so can use dx instead of the full
        grid array.

    drop_endpoints: boolean, optional (default False)
        Whether or not derivatives are computed for endpoints.
        If False, endpoints will be set to np.nan.
        Note that which points are endpoints depends on the method
        being used.

    Examples
    --------
    >>> import numpy as np
    >>> from pysindy.differentiation import FiniteDifference
    >>> t = np.linspace(0,1,5)
    >>> X = np.vstack((np.sin(t),np.cos(t))).T
    >>> fd = FiniteDifference()
    >>> fd._differentiate(X, t)
    array([[ 1.00114596,  0.00370551],
           [ 0.95885108, -0.24483488],
           [ 0.8684696 , -0.47444711],
           [ 0.72409089, -0.67456051],
           [ 0.53780339, -0.84443737]])
    """

    def __init__(self, order=2, d=1, is_uniform=False, drop_endpoints=False):
        if order <= 0 or not isinstance(order, int):
            raise ValueError("order must be a positive int")
        elif order > 2:
            raise NotImplementedError

        if d <= 0 or d > 4:
            raise ValueError("Derivative order must be " " 1, 2, or 3")

        if d > 1 and order != 2:
            raise ValueError(
                "For second or third order derivatives, order must equal 2 "
                "because only centered differences are implemented"
            )

        self.d = d
        self.order = order
        self.drop_endpoints = drop_endpoints
        self.is_uniform = is_uniform

    def _differentiate(self, x, t):
        """
        Apply finite difference method.
        """
        if self.order == 1:
            return self._forward_difference(x, t)
        else:
            return self._centered_difference(x, t)

    def _forward_difference(self, x, t=1):
        """
        First order forward difference
        (and 2nd order backward difference for final point).

        Note that in order to maintain compatibility with sklearn the,
        array returned, x_dot, always satisfies np.ndim(x_dot) == 2.
        """
        if self.is_uniform and not np.isscalar(t):
            t = t[1] - t[0]

        x_dot = np.full_like(x, fill_value=np.nan)

        # Uniform timestep (assume t contains dt)
        if np.isscalar(t):
            x_dot[:-1, :] = (x[1:, :] - x[:-1, :]) / t
            if not self.drop_endpoints:
                x_dot[-1, :] = (3 * x[-1, :] / 2 - 2 * x[-2, :] + x[-3, :] / 2) / t

        # Variable timestep
        else:
            t_diff = t[1:] - t[:-1]
            x_dot[:-1, :] = (x[1:, :] - x[:-1, :]) / t_diff[:, None]
            if not self.drop_endpoints:
                x_dot[-1, :] = (
                    3 * x[-1, :] / 2 - 2 * x[-2, :] + x[-3, :] / 2
                ) / t_diff[-1]

        return x_dot

    def _centered_difference(self, x, t=1, d=None):
        """
        Second order centered difference
        with third order forward/backward difference at endpoints.

        Warning: Sometimes has trouble with nonuniform grid spacing
        near boundaries

        Note that in order to maintain compatibility with sklearn the,
        array returned, x_dot, always satisfies np.ndim(x_dot) == 2.
        """
        if self.is_uniform and not np.isscalar(t):
            t = t[1] - t[0]

        if d is None:
            d = self.d

        x_dot = np.full_like(x, fill_value=np.nan)

        if d == 1:

            # Uniform timestep (assume t contains dt)
            if np.isscalar(t):
                x_dot[1:-1, :] = (x[2:, :] - x[:-2, :]) / (2 * t)
                if not self.drop_endpoints:
                    x_dot[0, :] = (
                        -11 / 6 * x[0, :] + 3 * x[1, :] - 3 / 2 * x[2, :] + x[3, :] / 3
                    ) / t
                    x_dot[-1, :] = (
                        11 / 6 * x[-1, :]
                        - 3 * x[-2, :]
                        + 3 / 2 * x[-3, :]
                        - x[-4, :] / 3
                    ) / t
                    # x_dot[0, :] = (-3.0 / 2 * x[0, :] +
                    #                2 * x[1, :] - x[2, :] / 2) / t
                    # x_dot[-1, :] = (
                    #     3.0 / 2 * x[-1, :] - 2 * x[-2, :] + x[-3, :] / 2
                    # ) / t

            # Variable timestep
            else:
                t_diff = t[2:] - t[:-2]
                x_dot[1:-1, :] = (x[2:, :] - x[:-2, :]) / t_diff[:, None]
                if not self.drop_endpoints:
                    x_dot[0, :] = (
                        -11 / 6 * x[0, :] + 3 * x[1, :] - 3 / 2 * x[2, :] + x[3, :] / 3
                    ) / (t_diff[0] / 2)
                    x_dot[-1, :] = (
                        11 / 6 * x[-1, :]
                        - 3 * x[-2, :]
                        + 3 / 2 * x[-3, :]
                        - x[-4, :] / 3
                    ) / (t_diff[-1] / 2)
                    # x_dot[0, :] = (-3.0 / 2 * x[0, :] +
                    #                 2 * x[1, :] - x[2, :] / 2) / (
                    #     t_diff[0] / 2
                    # )
                    # x_dot[-1, :] = (
                    #     3.0 / 2 * x[-1, :] - 2 * x[-2, :] + x[-3, :] / 2
                    # ) / (t_diff[0] / 2)

        if d == 2:

            # Uniform timestep (assume t contains dt)
            if np.isscalar(t):
                x_dot[1:-1, :] = (x[2:, :] - 2 * x[1:-1, :] + x[:-2, :]) / (t ** 2)
                if not self.drop_endpoints:
                    x_dot[0, :] = (
                        2 * x[0, :] - 5 * x[1, :] + 4 * x[2, :] - x[3, :]
                    ) / (t ** 2)
                    x_dot[-1, :] = (
                        2 * x[-1, :] - 5 * x[-2, :] + 4 * x[-3, :] - x[-4, :]
                    ) / (t ** 2)

            # Variable timestep
            else:
                t_diff = t[2:] - t[:-2]
                x_dot[1:-1, :] = (x[2:, :] - 2 * x[1:-1, :] + x[:-2, :]) / (
                    (t_diff[:, None] / 2.0) ** 2
                )
                if not self.drop_endpoints:
                    x_dot[0, :] = (
                        2 * x[0, :] - 5 * x[1, :] + 4 * x[2, :] - x[3, :]
                    ) / ((t_diff[0] / 2.0) ** 2)
                    x_dot[-1, :] = (
                        2 * x[-1, :] - 5 * x[-2, :] + 4 * x[-3, :] - x[-4, :]
                    ) / ((t_diff[-1] / 2.0) ** 2)

        if d == 3:

            # Uniform timestep (assume t contains dt)
            if np.isscalar(t):
                x_dot[2:-2, :] = (
                    x[4:, :] / 2.0 - x[3:-1, :] + x[1:-3, :] - x[:-4, :] / 2.0
                ) / (t ** 3)
                if not self.drop_endpoints:
                    x_dot[0, :] = (
                        -2.5 * x[0, :]
                        + 9 * x[1, :]
                        - 12 * x[2, :]
                        + 7 * x[3, :]
                        - 1.5 * x[4, :]
                    ) / (t ** 3)
                    x_dot[1, :] = (
                        -2.5 * x[1, :]
                        + 9 * x[2, :]
                        - 12 * x[3, :]
                        + 7 * x[4, :]
                        - 1.5 * x[5, :]
                    ) / (t ** 3)
                    x_dot[-1, :] = (
                        2.5 * x[-1, :]
                        - 9 * x[-2, :]
                        + 12 * x[-3, :]
                        - 7 * x[-4, :]
                        + 1.5 * x[-5, :]
                    ) / (t ** 3)
                    x_dot[-2, :] = (
                        2.5 * x[-2, :]
                        - 9 * x[-3, :]
                        + 12 * x[-4, :]
                        - 7 * x[-5, :]
                        + 1.5 * x[-6, :]
                    ) / (t ** 3)

            # Variable timestep
            else:
                t_diff = t[4:] - t[:-4]
                x_dot[2:-2, :] = (
                    x[4:, :] / 2.0 - x[3:-1, :] + x[1:-3, :] - x[:-4, :] / 2.0
                ) / ((t_diff[:, None] / 2.0) ** 3)
                if not self.drop_endpoints:
                    x_dot[0, :] = (
                        -2.5 * x[0, :]
                        + 9 * x[1, :]
                        - 12 * x[2, :]
                        + 7 * x[3, :]
                        - 1.5 * x[4, :]
                    ) / ((t_diff[0, None] / 2.0) ** 3)
                    x_dot[1, :] = (
                        -2.5 * x[1, :]
                        + 9 * x[2, :]
                        - 12 * x[3, :]
                        + 7 * x[4, :]
                        - 1.5 * x[5, :]
                    ) / ((t_diff[1, None] / 2.0) ** 3)
                    x_dot[-1, :] = (
                        2.5 * x[-1, :]
                        - 9 * x[-2, :]
                        + 12 * x[-3, :]
                        - 7 * x[-4, :]
                        + 1.5 * x[-5, :]
                    ) / ((t_diff[-1, None] / 2.0) ** 3)
                    x_dot[-2, :] = (
                        2.5 * x[-2, :]
                        - 9 * x[-3, :]
                        + 12 * x[-4, :]
                        - 7 * x[-5, :]
                        + 1.5 * x[-6, :]
                    ) / ((t_diff[-2, None] / 2.0) ** 3)

        if d > 3:
            return self._centered_difference(
                self._centered_difference(x, t, d=3), t, d=self.d - 3
            )

        return x_dot
