import numpy as np


# Linear, damped harmonic oscillator
def linear_damped_SHO(x, t):
    return [-0.1 * x[0] + 2 * x[1], -2 * x[0] - 0.1 * x[1]]


# Cubic, damped harmonic oscillator
def cubic_damped_SHO(x, t):
    return [
        -0.1 * x[0] ** 3 + 2 * x[1] ** 3,
        -2 * x[0] ** 3 - 0.1 * x[1] ** 3,
    ]


# Linear 3D toy system
def linear_3D(x, t):
    return [-0.1 * x[0] + 2 * x[1], -2 * x[0] - 0.1 * x[1], -0.3 * x[2]]


# Van der Pol ODE
def van_der_pol(x, t, p=[0.5]):
    return [x[1], p[0] * (1 - x[0] ** 2) * x[1] - x[0]]


# Duffing ODE
def duffing(x, t, p=[0.2, 0.05, 1]):
    return [x[1], -p[0] * x[1] - p[1] * x[0] - p[2] * x[0] ** 3]


# Lotka model
def lotka(x, t, p=[1, 10]):
    return [p[0] * x[0] - p[1] * x[0] * x[1], p[1] * x[0] * x[1] - 2 * p[0] * x[1]]


# Generic cubic oscillator model
def cubic_oscillator(x, t, p=[-0.1, 2, -2, -0.1]):
    return [p[0] * x[0] ** 3 + p[1] * x[1] ** 3, p[2] * x[0] ** 3 + p[3] * x[1] ** 3]


# Rossler model
def rossler(x, t, p=[0.2, 0.2, 5.7]):
    return [-x[1] - x[2], x[0] + p[0] * x[1], p[1] + (x[0] - p[2]) * x[2]]


# Hopf bifurcation model
def hopf(x, t, mu=-0.05, omega=1, A=1):
    return [
        mu * x[0] - omega * x[1] - A * x[0] * (x[0] ** 2 + x[1] ** 2),
        omega * x[0] + mu * x[1] - A * x[1] * (x[0] ** 2 + x[1] ** 2),
    ]


# Lorenz model
def lorenz(x, t, sigma=10, beta=2.66667, rho=28):
    return [
        sigma * (x[1] - x[0]),
        x[0] * (rho - x[2]) - x[1],
        x[0] * x[1] - beta * x[2],
    ]


# Control input for Lorenz + control
def lorenz_u(t):
    return np.column_stack([np.sin(2 * t) ** 2, t ** 2])


# Lorenz equations with control input
def lorenz_control(x, t, sigma=10, beta=2.66667, rho=28):
    u = lorenz_u(t)
    return [
        sigma * (x[1] - x[0]) + u[0, 0],
        x[0] * (rho - x[2]) - x[1],
        x[0] * x[1] - beta * x[2] - u[0, 1],
    ]


# Mean field model from Noack et al. 2003
def meanfield(x, t, mu=0.01):
    return [
        mu * x[0] - x[1] - x[0] * x[2],
        mu * x[1] + x[0] - x[1] * x[2],
        -x[2] + x[0] ** 2 + x[1] ** 2,
    ]


# Atmospheric oscillator from Tuwankotta et al and Trapping SINDy paper
def oscillator(x, t, mu1=0.05, mu2=-0.01, omega=3.0, alpha=-2.0, beta=-5.0, sigma=1.1):
    return [
        mu1 * x[0] + sigma * x[0] * x[1],
        mu2 * x[1] + (omega + alpha * x[1] + beta * x[2]) * x[2] - sigma * x[0] ** 2,
        mu2 * x[2] - (omega + alpha * x[1] + beta * x[2]) * x[1],
    ]


# Carbone and Veltri triadic MHD model
def mhd(x, t, nu=0.0, mu=0.0, sigma=0.0):
    return [
        -2 * nu * x[0] + 4.0 * (x[1] * x[2] - x[4] * x[5]),
        -5 * nu * x[1] - 7.0 * (x[0] * x[2] - x[3] * x[5]),
        -9 * nu * x[2] + 3.0 * (x[0] * x[1] - x[3] * x[4]),
        -2 * mu * x[4] + 2.0 * (x[5] * x[1] - x[2] * x[4]),
        -5 * mu * x[4] + sigma * x[5] + 5.0 * (x[2] * x[3] - x[0] * x[5]),
        -9 * mu * x[5] + sigma * x[4] + 9.0 * (x[4] * x[0] - x[1] * x[3]),
    ]


# Galerkin coefficients for the Burgers' equation in Noack et al. 2008
def burgers_galerkin(sigma=0.1, nu=0.025, U=1.0):
    r = 10
    L = np.zeros([r, r])

    for i in range(r // 2):
        # Dissipation
        L[2 * i, 2 * i] = -nu * (i + 1) ** 2
        L[2 * i + 1, 2 * i + 1] = -nu * (i + 1) ** 2

        # Mean flow advection
        L[2 * i, 2 * i + 1] = -(i + 1) * U
        L[2 * i + 1, 2 * i] = (i + 1) * U

    # Add forcing
    L[0, 0] += sigma
    L[1, 1] += sigma

    Q = np.zeros((r, r, r))
    Q[0, :, :] = np.array(
        [
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [-1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, -1, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, -1, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, -1, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, -1, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, -1, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, -1, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, -1, 0, 0],
        ]
    )

    Q[0, :, :] = 0.5 * (Q[0, :, :] + Q[0, :, :].T)

    Q[1, :, :] = np.array(
        [
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
            [-1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
            [0, 0, -1, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
            [0, 0, 0, 0, -1, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0, 0, -1, 0, 0, 0],
        ]
    )

    Q[1, :, :] = 0.5 * (Q[1, :, :] + Q[1, :, :].T)

    Q[2, :, :] = np.array(
        [
            [1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, -1, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [-2, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, -2, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, -2, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, -2, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, -2, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, -2, 0, 0, 0, 0],
        ]
    )

    Q[2, :, :] = 0.5 * (Q[2, :, :] + Q[2, :, :].T)

    Q[3, :, :] = np.array(
        [
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [2, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 2, 0, 0, 0, 0, 0, 0, 0, 0],
            [-2, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 2, 0, 0, 0, 0, 0, 0],
            [0, 0, -2, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 2, 0, 0, 0, 0],
            [0, 0, 0, 0, -2, 0, 0, 0, 0, 0],
        ]
    )

    Q[3, :, :] = 0.5 * (Q[3, :, :] + Q[3, :, :].T)

    Q[4, :, :] = np.array(
        [
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [3, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, -3, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [-3, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, -3, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, -3, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, -3, 0, 0, 0, 0, 0, 0],
        ]
    )

    Q[4, :, :] = 0.5 * (Q[4, :, :] + Q[4, :, :].T)

    Q[5, :, :] = np.array(
        [
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 3, 0, 0, 0, 0, 0, 0, 0, 0],
            [3, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 3, 0, 0, 0, 0, 0, 0, 0, 0],
            [-3, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 3, 0, 0, 0, 0, 0, 0],
            [0, 0, -3, 0, 0, 0, 0, 0, 0, 0],
        ]
    )

    Q[5, :, :] = 0.5 * (Q[5, :, :] + Q[5, :, :].T)

    Q[6, :, :] = np.array(
        [
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 2, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, -2, 0, 0, 0, 0, 0, 0],
            [4, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, -4, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [-4, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, -4, 0, 0, 0, 0, 0, 0, 0, 0],
        ]
    )

    Q[6, :, :] = 0.5 * (Q[6, :, :] + Q[6, :, :].T)

    Q[7, :, :] = np.array(
        [
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 4, 0, 0, 0, 0, 0, 0, 0],
            [0, 4, 0, 0, 0, 0, 0, 0, 0, 0],
            [4, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 4, 0, 0, 0, 0, 0, 0, 0, 0],
            [-4, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        ]
    )

    Q[7, :, :] = 0.5 * (Q[7, :, :] + Q[7, :, :].T)

    Q[8, :, :] = np.array(
        [
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 5, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, -5, 0, 0, 0, 0, 0, 0],
            [5, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, -5, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        ]
    )

    Q[8, :, :] = 0.5 * (Q[8, :, :] + Q[8, :, :].T)

    Q[9, :, :] = np.array(
        [
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 5, 0, 0, 0, 0, 0, 0],
            [0, 0, 5, 0, 0, 0, 0, 0, 0, 0],
            [0, 5, 0, 0, 0, 0, 0, 0, 0, 0],
            [5, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        ]
    )

    Q[9, :, :] = 0.5 * (Q[9, :, :] + Q[9, :, :].T)

    Q = Q / np.sqrt(np.pi)

    # Check for conservation in nonlinearity
    tol = 1e-15  # Numerical precision
    for i in range(Q.shape[0]):
        for j in range(Q.shape[0]):
            for k in range(Q.shape[0]):
                perm_sum = (
                    Q[i, j, k]
                    + Q[i, k, j]
                    + Q[j, i, k]
                    + Q[j, k, i]
                    + Q[k, i, j]
                    + Q[k, j, i]
                )
                assert perm_sum < tol
    return L, Q


# Below this line are models only suitable for SINDy-PI, since they are implicit #
# and therefore require a library Theta(X, Xdot) rather than just Theta(X) #

# Michaelis–Menten model for enzyme kinetics
def enzyme(x, t, jx=0.6, Vmax=1.5, Km=0.3):
    return jx - Vmax * x / (Km + x)


# Bacterial competence system (Mangan et al. 2016)
def bacterial(x, t, a1=0.004, a2=0.07, a3=0.04, b1=0.82, b2=1854.5):
    return [
        a1 + a2 * x[0] ** 2 / (a3 + x[0] ** 2) - x[0] / (1 + x[0] + x[1]),
        b1 / (1 + b2 * x[0] ** 5) - x[1] / (1 + x[0] + x[1]),
    ]


# yeast glycolysis model, note that there are many typos in the sindy-pi paper
def yeast(
    x,
    t,
    c1=2.5,
    c2=-100,
    c3=13.6769,
    d1=200,
    d2=13.6769,
    d3=-6,
    d4=-6,
    e1=6,
    e2=-64,
    e3=6,
    e4=16,
    f1=64,
    f2=-13,
    f3=13,
    f4=-16,
    f5=-100,
    g1=1.3,
    g2=-3.1,
    h1=-200,
    h2=13.6769,
    h3=128,
    h4=-1.28,
    h5=-32,
    j1=6,
    j2=-18,
    j3=-100,
):
    return [
        c1 + c2 * x[0] * x[5] / (1 + c3 * x[5] ** 4),
        d1 * x[0] * x[5] / (1 + d2 * x[5] ** 4) + d3 * x[1] - d4 * x[1] * x[6],
        e1 * x[1] + e2 * x[2] + e3 * x[1] * x[6] + e4 * x[2] * x[5],
        f1 * x[2] + f2 * x[3] + f3 * x[4] + f4 * x[2] * x[5] + f5 * x[3] * x[6],
        g1 * x[3] + g2 * x[4],
        h3 * x[2]
        + h5 * x[5]
        + h4 * x[2] * x[6]
        + h1 * x[0] * x[5] / (1 + h2 * x[5] ** 4),
        j1 * x[1] + j2 * x[1] * x[6] + j3 * x[3] * x[6],
    ]


# Cart on a pendulum
def pendulum_on_cart(x, t, m=1, M=1, L=1, F=0, g=9.81):
    return [
        x[2],
        x[3],
        (
            (M + m) * g * np.sin(x[0])
            - F * np.cos(x[0])
            - m * L * np.sin(x[0]) * np.cos(x[0]) * x[2] ** 2
        )
        / (L * (M + m * np.sin(x[0]) ** 2)),
        (m * L * np.sin(x[0]) * x[2] ** 2 + F - m * g * np.sin(x[0]) * np.cos(x[0]))
        / (M + m * np.sin(x[0]) ** 2),
    ]


# Control input models for kinematic single-track model
def f_steer(
    x,
    u,
    min_sangle=-0.91,
    max_sangle=0.91,
    min_svel=-0.4,
    max_svel=0.4,
    min_vel=-13.9,
    max_vel=45.8,
    switch_vel=4.755,
    amax=11.5,
):
    return 0


def f_acc(
    y,
    u,
    min_sangle=-0.91,
    max_sangle=0.91,
    min_svel=-0.4,
    max_svel=0.4,
    min_vel=-13.9,
    max_vel=45.8,
    switch_vel=4.755,
    amax=11.5,
):
    return 0


# CommonRoad kinematic single-track model
def kinematic_commonroad(x, t, u, amax=11.5, lwb=2.391):
    return [
        x[3] * np.cos(x[4]),
        x[3] * np.sin(x[4]),
        f_steer(x[0], u[0]),
        f_acc(x[1], u[1]),
        x[1] * np.tan(x[0]) / lwb,
    ]


# Infamous double pendulum problem (frictionless if k1=k2=0)
def double_pendulum(
    x,
    t,
    m1=0.2704,
    m2=0.2056,
    a1=0.191,
    a2=0.1621,
    L1=0.2667,
    L2=0.2667,
    I1=0.003,
    I2=0.0011,
    g=9.81,
    k1=0,
    k2=0,
):
    return [
        x[2],
        x[3],
        (
            L1 * a2 ** 2 * g * m2 ** 2 * np.sin(x[0])
            - 2 * L1 * a2 ** 3 * x[3] ** 2 * m2 ** 2 * np.sin(x[0] - x[1])
            + 2 * I2 * L1 * g * m2 * np.sin(x[0])
            + L1 * a2 ** 2 * g * m2 ** 2 * np.sin(x[0] - 2 * x[1])
            + 2 * I2 * a1 * g * m1 * np.sin(x[0])
            - (L1 * a2 * x[2] * m2) ** 2 * np.sin(2 * (x[0] - x[1]))
            - 2 * I2 * L1 * a2 * x[3] ** 2 * m2 * np.sin(x[0] - x[1])
            + 2 * a1 * a2 ** 2 * g * m1 * m2 * np.sin(x[0])
        )
        / (
            2 * I1 * I2
            + (L1 * a2 * m2) ** 2
            + 2 * I2 * L1 ** 2 * m2
            + 2 * I2 * a1 ** 2 * m1
            + 2 * I1 * a2 ** 2 * m2
            - (L1 * a2 * m2) ** 2 * np.cos(2 * (x[0] - x[1]))
            + 2 * (a1 * a2) ** 2 * m1 * m2
        ),
        (
            a2
            * m2
            * (
                2 * I1 * g * np.sin(x[1])
                + 2 * L1 ** 3 * x[2] ** 2 * m2 * np.sin(x[0] - x[1])
                + 2 * L1 ** 2 * g * m2 * np.sin(x[1])
                + 2 * I1 * L1 * x[2] ** 2 * np.sin(x[0] - x[1])
                + 2 * a1 ** 2 * g * m1 * np.sin(x[1])
                + L1 ** 2 * a2 * x[3] ** 2 * m2 * np.sin(*(x[0] - x[1]))
                + 2 * L1 * a1 ** 2 * x[2] ** 2 * m1 * np.sin(x[0] - x[1])
                - 2 * L1 ** 2 * g * m2 * np.cos(x[0] - x[1]) * np.sin(x[0])
                - 2 * L1 * a1 * g * m1 * np.cos(x[0] - x[1]) * np.sin(x[0])
            )
        )
        / (
            2
            * (
                I1 * I2
                + (L1 * a2 * m2) ** 2
                + I2 * L1 ** 2 * m2
                + I2 * a1 ** 2 * m1
                + I1 * a2 ** 2 * m2
                - (L1 * a2 * m2) ** 2 * np.cos(x[0] - x[1]) ** 2
                + a1 ** 2 * a2 ** 2 * m1 * m2
            )
        ),
    ]
