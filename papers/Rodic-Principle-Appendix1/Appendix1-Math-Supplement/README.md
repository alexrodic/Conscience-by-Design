# Appendix 1 – The Rodić Principle: Mathematical Supplement

This folder contains the reproducible numerical implementation that accompanies:

**Appendix 1 – The Rodić Principle: Mathematical Supplement (v1.0)**  
Author: Aleksandar Rodić  
License: CC BY 4.0 (text) / MIT (code)

## Contents

- `rodic_principle_math_supplement.py`  
  Reference implementation of the Rodić Principle dynamical model.  
  It:
  - defines the moral state \(M \in \mathbb{R}^3\) and equilibrium \(M^* = (1,1,1)^T\),
  - specifies the matrices \(Q, P\) and the linearization \(A = Df(M^*) - Q\),
  - implements the deterministic dynamics \(dM/dt = F(M)\),
  - implements the 3D Ornstein–Uhlenbeck process \(dZ = A Z dt + \Sigma dW_t\),
  - checks the global stability condition from Theorem 6.1,
  - verifies that \(A\) is Hurwitz,
  - computes deterministic and stochastic half-lives,
  - solves the continuous Lyapunov equation for the stationary covariance,
  - generates Figures 1–7 as `.png` files.

- `tests/test_rodic_principle.py`  
  Pytest-based unit tests covering:
  - equilibrium properties of \(V(M)\) and \(E(M)\),
  - the global stability inequality,
  - Hurwitz property of \(A\),
  - deterministic convergence to \(M^*\),
  - OU simulation sanity checks,
  - stochastic half-life distribution bounds,
  - properties of the stationary covariance,
  - sign of \(dV/dt\) away from equilibrium.

- `requirements.txt`  
  Minimal Python dependencies.

- `Makefile` (optional)  
  Convenience targets for generating figures and running tests.

## Requirements

- Python 3.9+ (recommended)
- NumPy
- SciPy
- Matplotlib
- Pytest (for running the test suite)

Install all dependencies with:

```bash
pip install -r requirements.txt

How to reproduce Figures 1–7

From this folder, run:
python rodic_principle_math_supplement.py

This will produce the following files in the current directory:

fig1_deterministic_convergence.png

fig2_potential_levels.png

fig3_lyapunov_decay.png

fig4_eigenvalues.png

fig5_ou_trajectories.png

fig6_halflife_hist.png

fig7_det_vs_sto.png

The numerical results and plots correspond to Figures 1–7 in the Appendix.

How to run the tests

From this folder (or from the project root):

pytest -q


To check coverage (optional):

pytest --cov=rodic_principle_math_supplement


All tests should pass, and coverage should be high (close to or above 90%).

License

Text and documentation: Creative Commons Attribution 4.0 (CC BY 4.0)

Source code and implementations: MIT License

© 2025 Aleksandar Rodić — Conscience by Design Initiative.

## requirements.txt

`Appendix1-Math-Supplement/requirements.txt`:

```text
numpy>=1.24
scipy>=1.10
matplotlib>=3.7
pytest>=7.0
pytest-cov>=4.0
