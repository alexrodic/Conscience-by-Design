#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
The Rodić Principle — Mathematical Supplement (Reference Implementation)
Author: Aleksandar Rodić
License: MIT (code) / CC BY 4.0 (text)

This script reproduces the core numerical results and figures (Fig. 1–7)
described in:

Appendix 1 – The Rodić Principle: Mathematical Supplement (v1.0)

It defines:
    • Moral state M and equilibrium M*
    • Matrices Q, P, and linearization A
    • Deterministic nonlinear dynamics
    • Linearized Ornstein–Uhlenbeck process
    • Stability diagnostics (Theorem 6.1, Hurwitz condition)
    • Numerical generation of Figures 1–7
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import eigvals
from scipy.linalg import solve_continuous_lyapunov

# =============================================================================
# 1. Global parameters and core objects
# =============================================================================

# Deterministic random seed for full reproducibility
RANDOM_SEED: int = 42
rng = np.random.default_rng(RANDOM_SEED)

# Equilibrium point M* = (1, 1, 1)^T
M_STAR: np.ndarray = np.array([1.0, 1.0, 1.0])

# Positive definite matrices Q and P
Q: np.ndarray = np.diag([1.0, 1.2, 1.5])  # restoring potential matrix (strictly convex)
P: np.ndarray = np.eye(3)                 # Lyapunov weighting (for simplicity P = I)

# Linear drift matrix B used in f(M) = B (M - M*)
B: np.ndarray = np.array([
    [-0.1,  0.05, 0.00],
    [ 0.00, -0.1, 0.05],
    [ 0.00,  0.00, -0.1],
])

# Lipschitz constant for f; here we choose a simple globally Lipschitz drift
L: float = 0.5

# Time discretization for simulations
T_FINAL: float = 10.0       # final time
DT: float = 0.01            # time step
N_STEPS: int = int(T_FINAL / DT)


# =============================================================================
# 2. Drift field and full deterministic dynamics
# =============================================================================

def drift_f(M: np.ndarray) -> np.ndarray:
    """
    Drift field f(M).

    We choose a linear, globally Lipschitz drift centered at M*:
        f(M) = B (M - M*)
    where B has small norm so that the restoring term -Q (M - M*)
    dominates in the full dynamics.
    """
    return B @ (M - M_STAR)


def F(M: np.ndarray) -> np.ndarray:
    """
    Full nonlinear vector field:
        dM/dt = F(M) = f(M) - Q (M - M*)
    """
    return drift_f(M) - Q @ (M - M_STAR)


# =============================================================================
# 3. Lyapunov function and stability diagnostics
# =============================================================================

def V(M: np.ndarray) -> float:
    """
    Quadratic Lyapunov function:
        V(M) = (M - M*)^T P (M - M*)
    """
    x = M - M_STAR
    return float(x.T @ P @ x)


def dV_dt(M: np.ndarray) -> float:
    """
    Time derivative of V along trajectories of the full system:
        dV/dt = 2 (M - M*)^T P F(M)

    For P = I this reduces to:
        dV/dt = 2 (M - M*)^T F(M)
    """
    x = M - M_STAR
    return float(2.0 * x.T @ P @ F(M))


def check_global_stability_condition() -> bool:
    """
    Check the key inequality from Theorem 6.1:
        lambda_min(PQ) > L * lambda_max(P)

    Returns
    -------
    bool
        True if the inequality holds, False otherwise.
    """
    PQ = P @ Q
    eigvals_PQ = np.linalg.eigvals(PQ)
    eigvals_P = np.linalg.eigvals(P)

    lam_min_PQ = np.min(np.real(eigvals_PQ))
    lam_max_P = np.max(np.real(eigvals_P))

    inequality_holds = lam_min_PQ > L * lam_max_P

    print("=== Global Stability Condition (Theorem 6.1) ===")
    print(f"lambda_min(PQ) = {lam_min_PQ:.4f}")
    print(f"L * lambda_max(P) = {L * lam_max_P:.4f}")
    print(f"Inequality holds: {inequality_holds}")
    print("================================================\n")

    return inequality_holds


# =============================================================================
# 4. Deterministic simulation (Fig. 1, Fig. 3, Fig. 7 - deterministic part)
# =============================================================================

def simulate_deterministic(M0: np.ndarray) -> np.ndarray:
    """
    Explicit Euler integration of the deterministic system:
        dM/dt = F(M)
    """
    M = np.zeros((N_STEPS + 1, 3), dtype=float)
    M[0] = M0
    for k in range(N_STEPS):
        M[k + 1] = M[k] + DT * F(M[k])
    return M


# =============================================================================
# 5. Convex potential E(M) and level sets (Fig. 2)
# =============================================================================

def E(M: np.ndarray) -> float:
    """
    Convex quadratic potential:
        E(M) = 0.5 * (M - M*)^T Q (M - M*)
    """
    x = M - M_STAR
    return float(0.5 * x.T @ Q @ x)


def plot_potential_levels() -> None:
    """
    Produce Fig. 2: level sets of E(M) on a 2D slice (fixing R = 1).
    """
    t_vals = np.linspace(0.0, 1.5, 100)
    h_vals = np.linspace(0.0, 1.5, 100)

    T_grid, H_grid = np.meshgrid(t_vals, h_vals, indexing="ij")
    E_grid = np.zeros_like(T_grid)

    for i in range(T_grid.shape[0]):
        for j in range(T_grid.shape[1]):
            M_ij = np.array([T_grid[i, j], H_grid[i, j], 1.0])  # fix R = 1
            E_grid[i, j] = E(M_ij)

    plt.figure()
    CS = plt.contour(T_grid, H_grid, E_grid, levels=15)
    plt.clabel(CS, inline=True, fontsize=8)
    plt.xlabel("T")
    plt.ylabel("H")
    plt.title("Fig. 2 – Level sets of the convex potential E(M)")
    plt.tight_layout()
    plt.savefig("fig2_potential_levels.png", dpi=300)
    plt.close()


# =============================================================================
# 6. Linearization, eigenvalues and half-life (Fig. 4)
# =============================================================================

def linearization_matrix_A() -> np.ndarray:
    """
    Linearization at M*:
        A = Df(M*) - Q

    For f(M) = B (M - M*), we have Df(M*) = B.
    """
    return B - Q


def compute_half_life(A: np.ndarray) -> tuple[float, np.ndarray]:
    """
    Compute deterministic half-life:
        tau_{1/2} = ln(2) / |Re(lambda_max(A))|
    where lambda_max(A) is the eigenvalue with largest real part (< 0).

    Returns
    -------
    (tau_half, lambdas)
        Deterministic half-life and array of eigenvalues.
    """
    lambdas = eigvals(A)
    real_parts = np.real(lambdas)
    lam_max_real = np.max(real_parts)  # largest (least negative) real part
    tau_half = np.log(2.0) / abs(lam_max_real)
    return float(tau_half), lambdas


def plot_eigenvalues(A: np.ndarray) -> None:
    """
    Produce Fig. 4: eigenvalues of A in the complex plane.
    """
    lambdas = eigvals(A)

    plt.figure()
    plt.scatter(np.real(lambdas), np.imag(lambdas))
    plt.axvline(0.0, linestyle="--", linewidth=1.0)
    plt.xlabel("Re(λ)")
    plt.ylabel("Im(λ)")
    plt.title("Fig. 4 – Eigenvalues of the Jacobian A")
    plt.tight_layout()
    plt.savefig("fig4_eigenvalues.png", dpi=300)
    plt.close()


# =============================================================================
# 7. Stochastic OU simulation (Fig. 5, Fig. 6, Fig. 7 - stochastic part)
# =============================================================================

def simulate_ou(
    A: np.ndarray,
    Sigma: np.ndarray,
    Z0: np.ndarray,
    t_final: float,
    dt: float,
) -> np.ndarray:
    """
    Simulate the 3D Ornstein–Uhlenbeck process:
        dZ = A Z dt + Sigma dW_t
    using Euler–Maruyama discretization.
    """
    n_steps = int(t_final / dt)
    Z = np.zeros((n_steps + 1, 3), dtype=float)
    Z[0] = Z0

    for k in range(n_steps):
        dW = rng.normal(loc=0.0, scale=np.sqrt(dt), size=3)
        Z[k + 1] = Z[k] + dt * (A @ Z[k]) + Sigma @ dW

    return Z


def estimate_half_life_single_path(
    A: np.ndarray,
    Sigma: np.ndarray,
    Z0: np.ndarray,
    dt: float,
    threshold: float = 0.5,
    t_final: float = T_FINAL,
) -> float:
    """
    Estimate a stochastic half-life for a single trajectory:
        time until ||Z(t)|| <= threshold * ||Z(0)||.
    """
    Z = Z0.copy()
    norm0 = np.linalg.norm(Z0)
    t = 0.0

    while t < t_final:
        dW = rng.normal(loc=0.0, scale=np.sqrt(dt), size=3)
        Z = Z + dt * (A @ Z) + Sigma @ dW
        t += dt
        if np.linalg.norm(Z) <= threshold * norm0:
            return t

    return t_final  # did not reach threshold within horizon


def monte_carlo_half_lives(
    A: np.ndarray,
    Sigma: np.ndarray,
    n_samples: int = 500,
    dt: float = DT,
) -> np.ndarray:
    """
    Monte Carlo estimation of stochastic half-lives (Fig. 6).
    """
    half_lives = np.zeros(n_samples, dtype=float)

    for i in range(n_samples):
        # random initial condition around M*
        Z0 = rng.normal(loc=0.5, scale=0.2, size=3)
        half_lives[i] = estimate_half_life_single_path(A, Sigma, Z0, dt)

    return half_lives


def plot_ou_sample_paths(A: np.ndarray, Sigma: np.ndarray) -> None:
    """
    Produce Fig. 5: sample paths of the OU process (norm of Z(t)).
    """
    t_grid = np.linspace(0.0, T_FINAL, N_STEPS + 1)

    # Use three different initial conditions
    Z0_list = [
        np.array([0.5, 0.0, 0.0]),
        np.array([0.0, 0.5, 0.0]),
        np.array([0.0, 0.0, 0.5]),
    ]

    plt.figure()
    for Z0 in Z0_list:
        Z = simulate_ou(A, Sigma, Z0, T_FINAL, DT)
        plt.plot(t_grid, np.linalg.norm(Z, axis=1))

    plt.xlabel("t")
    plt.ylabel("||Z(t)||")
    plt.title("Fig. 5 – Sample paths of the OU process (norm)")
    plt.tight_layout()
    plt.savefig("fig5_ou_trajectories.png", dpi=300)
    plt.close()

rodic_principle_math_supplement
def plot_half_life_histogram(half_lives: np.ndarray) -> None:
    """
    Produce Fig. 6: histogram of stochastic half-life times.
    """
    plt.figure()
    plt.hist(half_lives, bins=30, density=True)
    plt.xlabel("Stochastic half-life τ_{1/2}^{sto}")
    plt.ylabel("Density")
    plt.title("Fig. 6 – Histogram of stochastic half-lives (Monte Carlo)")
    plt.tight_layout()
    plt.savefig("fig6_halflife_hist.png", dpi=300)
    plt.close()


# =============================================================================
# 8. Comparison of deterministic vs stochastic decay (Fig. 1, Fig. 3, Fig. 7)
# =============================================================================

def plot_deterministic_convergence(M_traj: np.ndarray) -> None:
    """
    Produce Fig. 1 and Fig. 3 (Lyapunov decay for a single deterministic path).
    """
    t_grid = np.linspace(0.0, T_FINAL, M_traj.shape[0])

    # Fig. 1 – deterministic trajectories of each component
    plt.figure()
    plt.plot(t_grid, M_traj[:, 0], label="T(t)")
    plt.plot(t_grid, M_traj[:, 1], label="H(t)")
    plt.plot(t_grid, M_traj[:, 2], label="R(t)")
    plt.axhline(1.0, linestyle="--", linewidth=1.0)
    plt.xlabel("t")
    plt.ylabel("Components")
    plt.title("Fig. 1 – Deterministic convergence to M* = (1,1,1)")
    plt.legend()
    plt.tight_layout()
    plt.savefig("fig1_deterministic_convergence.png", dpi=300)
    plt.close()

    # Fig. 3 – Lyapunov function decay V(t)
    V_vals = np.array([V(M_traj[k]) for k in range(M_traj.shape[0])])

    plt.figure()
    plt.semilogy(t_grid, V_vals)
    plt.xlabel("t")
    plt.ylabel("V(t)")
    plt.title("Fig. 3 – Lyapunov function decay")
    plt.tight_layout()
    plt.savefig("fig3_lyapunov_decay.png", dpi=300)
    plt.close()


def plot_det_vs_sto_decay(A: np.ndarray, Sigma: np.ndarray) -> None:
    """
    Produce Fig. 7: comparison of deterministic and stochastic decay of ||Z(t)||.
    """
    t_grid = np.linspace(0.0, T_FINAL, N_STEPS + 1)

    # Deterministic linear system dZ = A Z dt, starting from Z0
    Z0 = np.array([0.8, -0.4, 0.2])
    Z_det = np.zeros((N_STEPS + 1, 3), dtype=float)
    Z_det[0] = Z0
    for k in range(N_STEPS):
        Z_det[k + 1] = Z_det[k] + DT * (A @ Z_det[k])

    # Stochastic OU with the same A and Z0
    Z_sto = simulate_ou(A, Sigma, Z0, T_FINAL, DT)

    plt.figure()
    plt.plot(t_grid, np.linalg.norm(Z_det, axis=1), label="Deterministic")
    plt.plot(t_grid, np.linalg.norm(Z_sto, axis=1), label="Stochastic", alpha=0.8)
    plt.xlabel("t")
    plt.ylabel("||Z(t)||")
    plt.title("Fig. 7 – Deterministic vs stochastic decay")
    plt.legend()
    plt.tight_layout()
    plt.savefig("fig7_det_vs_sto.png", dpi=300)
    plt.close()


# =============================================================================
# 9. Stationary covariance (OU Lyapunov equation)
# =============================================================================

def compute_stationary_covariance(A: np.ndarray, Sigma: np.ndarray) -> np.ndarray:
    """
    Solve the continuous Lyapunov equation for the stationary covariance Γ:
        A Γ + Γ A^T + Σ Σ^T = 0
    """
    Q_lyap = -Sigma @ Sigma.T
    Gamma = solve_continuous_lyapunov(A, Q_lyap)
    return Gamma


# =============================================================================
# 10. Main execution block
# =============================================================================

if __name__ == "__main__":
    # 1) Check global asymptotic stability condition
    check_global_stability_condition()

    # 2) Linearization and spectral analysis
    A = linearization_matrix_A()
    tau_half_det, lambdas = compute_half_life(A)
    print("=== Spectral Analysis of A ===")
    print(f"Eigenvalues(A) = {lambdas}")
    print(f"Deterministic half-life τ_1/2^det = {tau_half_det:.4f}")
    print("A is Hurwitz:", np.all(np.real(lambdas) < 0.0))
    print("================================\n")

    # 3) Deterministic trajectory from a non-equilibrium initial state
    M0 = np.array([0.2, 1.3, 0.5])
    M_traj = simulate_deterministic(M0)
    plot_deterministic_convergence(M_traj)

    # 4) Potential level sets (Fig. 2)
    plot_potential_levels()

    # 5) Eigenvalues plot (Fig. 4)
    plot_eigenvalues(A)

    # 6) Stochastic OU process analysis
    sigma = 0.2
    Sigma = sigma * np.eye(3)

    # OU sample paths (Fig. 5)
    plot_ou_sample_paths(A, Sigma)

    # Monte Carlo half-lives (Fig. 6)
    half_lives = monte_carlo_half_lives(A, Sigma, n_samples=500, dt=DT)
    plot_half_life_histogram(half_lives)

    # 7) Deterministic vs stochastic decay (Fig. 7)
    plot_det_vs_sto_decay(A, Sigma)

    # 8) Stationary covariance of the OU process
    Gamma = compute_stationary_covariance(A, Sigma)
    print("=== Stationary covariance Γ of OU process ===")
    print(Gamma)
    print("=============================================")
