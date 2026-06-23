import numpy as np
from experiments import (
    run_double_descent_experiment,
    run_noise_experiment,
    run_gd_vs_closedform
)
from plots import plot_double_descent, plot_noise_comparison

#CONFIG
N_TRAIN    = 150
NOISE_STD  = 0.5
LAMBDA     = 0.5
SEED       = 42
# Sweep d from 1 to 3*n in steps of 5 (adjust step for speed vs resolution)
D_VALUES   = list(range(1, 3 * N_TRAIN + 1, 3))

#MAIN PART
print("=" * 60)
print("EXPERIMENT 1: Double Descent (LS vs Ridge)")
print("=" * 60)
results = run_double_descent_experiment(
    n_train=N_TRAIN,
    d_values=D_VALUES,
    noise_std=NOISE_STD,
    lam=LAMBDA,
    seed=SEED
)
# Change the save_path to this:
plot_double_descent(results, n_train=N_TRAIN, lam=LAMBDA,
                    save_path='double_descent.pdf')

#NOISE EXTENSION
print("\n" + "=" * 60)
print("EXPERIMENT 2: Effect of Noise")
print("=" * 60)
noise_results = run_noise_experiment(
    n_train=N_TRAIN,
    d_values=D_VALUES,
    noise_levels=[0.1, 0.5, 1.0, 2.0],
    seed=SEED
)
plot_noise_comparison(noise_results, n_train=N_TRAIN,
                      save_path='noise_comparison.pdf')

#GD VS CLOSED-FORM EXTENSION
print("\n" + "=" * 60)
print("EXPERIMENT 3: Gradient Descent vs Closed-Form (d < n)")
print("=" * 60)
run_gd_vs_closedform(n_train=100, d=50, noise_std=NOISE_STD, seed=SEED)

print("\nDone. Plots saved.")

# ─── DIAGNOSTIC OVERRIDE PRINT ───────────────────────────────────────────
print("\n" + "!" * 50)
print("CRITICAL DIAGNOSTIC CHECK (d=148, near threshold)")
print("!" * 50)
idx_148 = results['d'].index(148) if 148 in results['d'] else 0
if idx_148:
    print(f"LS TEST MSE AT d=148:    {results['ls_test'][idx_148]:.4f}")
    print(f"RIDGE TEST MSE AT d=148: {results['ridge_test'][idx_148]:.4f}")
else:
    print(f"LS TEST MSE AT FIRST STEP: {results['ls_test'][0]:.4f}")
print("!" * 50 + "\n")