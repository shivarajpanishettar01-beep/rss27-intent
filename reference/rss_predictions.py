"""Reproduce every number quoted in the RSS 2027 extended abstract's predictions.

Run:  python3 rss_predictions.py
All values are Monte-Carlo checks of closed-form results (2M samples, seed 0).
"""
import numpy as np

rng = np.random.default_rng(0)
N = 2_000_000
H = 1.0  # chunk horizon in seconds


def report(name, D):
    med = np.median(D)
    print(f"\n=== {name} ===")
    print(f"Eq.2 bound, median rule  E|D-med(D)| = {np.mean(abs(D - med)):.2f} s")
    T = rng.permutation(D)  # independent draw from p_D
    print(f"Calibrated imitator      E|T-D|, T~p_D = {np.mean(abs(T - D)):.2f} s")
    T = rng.exponential(D.mean(), N)  # constant hazard 1/E[D]
    print(f"Memoryless hedger        E|T-D|, T~Exp(E[D]) = {np.mean(abs(T - D)):.2f} s"
          f"  (leaves early in {np.mean(T < D):.0%} of holds)")
    pibar = np.mean(np.minimum(H, D)) / np.mean(D)
    print(f"Eq.3 pi_bar (H={H:.0f} s)      = {pibar:.3f}")
    for tau in (0.2, 0.4):
        print(f"  threshold-corrected rate, tau={tau} s: (H-tau)/E[D] = {(H - tau) / D.mean():.3f}")
    print("  hazard pi(e) = P(D <= e+H | D > e):",
          {e: round(float(np.mean(D[D > e] <= e + H)), 3) for e in (0, 1, 2, 4, 6, 8, 9) if np.any(D > e)})


report("Uniform holds, D ~ U[2,10] s", rng.uniform(2, 10, N))
report("Shifted-exponential holds, D = 2 + Exp(mean 4) s", 2 + rng.exponential(4.0, N))
