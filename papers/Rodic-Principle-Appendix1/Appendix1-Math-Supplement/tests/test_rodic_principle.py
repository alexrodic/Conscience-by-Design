import numpy as np
import rodic_principle_math_supplement as rpm


def test_equilibrium_lyapunov_zero():
    # V(M*) i E(M*) moraju biti nula
    assert rpm.V(rpm.M_STAR) == 0.0
    assert rpm.E(rpm.M_STAR) == 0.0


def test_global_stability_condition_holds():
    assert rpm.check_global_stability_condition() is True


def test_linearization_matrix_is_hurwitz():
    A = rpm.linearization_matrix_A()
    eigvals = np.linalg.eigvals(A)
    assert np.all(np.real(eigvals) < 0.0)


def test_deterministic_convergence_to_equilibrium():
    M0 = np.array([0.2, 1.3, 0.5])
    traj = rpm.simulate_deterministic(M0)
    # poslednji korak blizu M*
    assert np.allclose(traj[-1], rpm.M_STAR, atol=1e-2)


def test_ou_simulation_shape_and_finiteness():
    A = rpm.linearization_matrix_A()
    Sigma = 0.2 * np.eye(3)
    Z0 = np.array([0.5, 0.0, 0.0])

    Z = rpm.simulate_ou(A, Sigma, Z0, t_final=rpm.T_FINAL, dt=rpm.DT)
    assert Z.shape == (rpm.N_STEPS + 1, 3)
    assert np.all(np.isfinite(Z))


def test_monte_carlo_half_lives_range():
    A = rpm.linearization_matrix_A()
    Sigma = 0.2 * np.eye(3)

    half_lives = rpm.monte_carlo_half_lives(
        A, Sigma, n_samples=10, dt=rpm.DT
    )
    # Svi half-life-ovi moraju biti u [0, T_FINAL]
    assert np.all(half_lives >= 0.0)
    assert np.all(half_lives <= rpm.T_FINAL)


def test_stationary_covariance_properties():
    A = rpm.linearization_matrix_A()
    Sigma = 0.2 * np.eye(3)
    Gamma = rpm.compute_stationary_covariance(A, Sigma)

    # Gamma mora biti simetrična
    assert np.allclose(Gamma, Gamma.T, atol=1e-10)

    # Dijagonala pozitivna (pozitivna semidefinitnost)
    assert np.all(np.diag(Gamma) > 0.0)


def test_dV_dt_negative_away_from_equilibrium():
    # Izaberi par tačaka dalje od ravnoteže
    samples = [
        np.array([0.0, 0.0, 0.0]),
        np.array([1.5, 0.5, 0.5]),
        np.array([0.5, 1.5, 1.5]),
    ]
    for M in samples:
        # dV/dt bi trebao biti <= 0 (za stabilan sistem)
        assert rpm.dV_dt(M) <= 0.0
