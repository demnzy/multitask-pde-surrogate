"""
Command-line runner for 2D Multi-Fidelity Heat Diffusion PDE Surrogate and Active Learning.

Usage:
    python run_benchmark.py --n-coarse 8 --n-fine 2 --n-active 5 --save-plots
"""

import argparse
import os
import sys
import numpy as np
import torch
import matplotlib.pyplot as plt

# Ensure local package import
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from pde_surrogate import (
    simulate_heat,
    get_peak_temperature,
    generate_random_heaters,
    fit_gp_surrogate,
    predict_fidelity,
    run_active_learning,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Multi-Fidelity Gaussian Process Surrogate & Active Learning Benchmark",
    )
    parser.add_argument("--n-coarse", type=int, default=8, help="Initial coarse (m=0) samples")
    parser.add_argument("--n-fine", type=int, default=2, help="Initial fine (m=1) samples")
    parser.add_argument("--n-active", type=int, default=5, help="Active learning query steps")
    parser.add_argument("--epochs", type=int, default=80, help="Initial GP training epochs")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--output-dir", type=str, default="figures", help="Directory to save figures")
    parser.add_argument("--save-plots", action="store_true", help="Save convergence and parity plots")
    return parser.parse_args()


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    print("=" * 68)
    print(" 2D Multi-Fidelity Heat Diffusion PDE Surrogate Benchmark")
    print("=" * 68)
    print(f"Initial Dataset: {args.n_coarse} coarse (16x16) + {args.n_fine} fine (64x64)")
    print(f"Active Learning Steps: {args.n_active} | Random Seed: {args.seed}\n")

    # 1. Physics solver sanity benchmark
    ref_heaters = torch.tensor([[0.3, 0.4, 10.0], [0.7, 0.7, 8.0]], dtype=torch.float32)
    peak_c, time_c = get_peak_temperature(ref_heaters, fidelity=0)
    peak_f, time_f = get_peak_temperature(ref_heaters, fidelity=1)

    speedup = time_f / max(time_c, 1e-6)
    print("--- 1. PDE Solver Runtime Benchmark ---")
    print(f"Low-Fidelity  (16x16,  40 steps): {peak_c:.4f} °C | {time_c * 1000:.2f} ms")
    print(f"High-Fidelity (64x64, 400 steps): {peak_f:.4f} °C | {time_f * 1000:.2f} ms")
    print(f"Fine solver measured computational cost factor: {speedup:.1f}x\n")

    # 2. Build initial training set
    print("--- 2. Generating Initial Training Samples ---")
    x_rows = []
    y_rows = []

    for i in range(args.n_coarse):
        setup = generate_random_heaters(2)
        peak, _ = get_peak_temperature(setup, fidelity=0)
        x_rows.append(torch.cat([setup.flatten(), torch.tensor([0.0])]))
        y_rows.append(peak)

    for i in range(args.n_fine):
        setup = generate_random_heaters(2)
        peak, _ = get_peak_temperature(setup, fidelity=1)
        x_rows.append(torch.cat([setup.flatten(), torch.tensor([1.0])]))
        y_rows.append(peak)

    train_x = torch.stack(x_rows)
    train_y = torch.tensor(y_rows, dtype=torch.float32)
    print(f"Generated {len(train_x)} training samples. Shape: {train_x.shape}\n")

    # 3. Train multi-fidelity GP
    print("--- 3. Optimizing Gaussian Process Surrogate ---")
    model, likelihood, losses = fit_gp_surrogate(train_x, train_y, num_epochs=args.epochs, lr=0.1, verbose=True)
    print(f"Final Negative MLL Loss: {losses[-1]:.4f}\n")

    # 4. Held-out validation check
    print("--- 4. Held-Out Generalization Evaluation ---")
    test_heaters = generate_random_heaters(2)
    pred_mean, pred_std = predict_fidelity(model, likelihood, test_heaters.flatten(), fidelity=1)
    actual_peak, _ = get_peak_temperature(test_heaters, fidelity=1)
    abs_err = abs(pred_mean.item() - actual_peak)

    print(f"Predicted HF Peak: {pred_mean.item():.4f} °C ± {2 * pred_std.item():.4f} °C (95% CI)")
    print(f"Ground Truth HF:   {actual_peak:.4f} °C")
    print(f"Discrepancy:       {abs_err:.4f} °C\n")

    # 5. Cost-Aware Active Learning Loop
    print("--- 5. Running Cost-Aware Active Learning Loop ---")
    final_model, final_lik, final_x, final_y, history = run_active_learning(
        train_x,
        train_y,
        n_iterations=args.n_active,
        pool_size=20,
        cost_low=time_c * 1000.0,
        cost_high=time_f * 1000.0,
        retrain_epochs=40,
        lr=0.08,
        verbose=True,
    )

    print("\n--- Summary Table ---")
    print(f"{'Iter':<5} | {'Fidelity':<10} | {'Cumul Time (ms)':<16} | {'Holdout Variance':<18}")
    print("-" * 55)
    for it, fid, tm, var in zip(
        history["iteration"],
        history["selected_fidelity"],
        history["cumulative_time_ms"],
        history["holdout_variance"],
    ):
        fid_disp = str(fid) if fid is not None else "Initial"
        print(f"{it:<5} | {fid_disp:<10} | {tm:<16.1f} | {var:<18.5f}")

    init_v = history["holdout_variance"][0]
    final_v = history["holdout_variance"][-1]
    reduction = ((init_v - final_v) / init_v) * 100.0 if init_v > 0 else 0.0
    print("-" * 55)
    print(f"Total variance reduction achieved: {reduction:.1f}%")
    print(f"Total active query compute spent:  {history['cumulative_time_ms'][-1]:.1f} ms\n")

    # 6. Optional Plotting
    if args.save_plots:
        os.makedirs(args.output_dir, exist_ok=True)
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(7, 6), sharex=True)
        ax1.plot(history["iteration"], history["holdout_variance"], marker="o", color="#2ca02c", lw=2)
        ax1.set_ylabel("Holdout Predictive Variance")
        ax1.set_title("Active Learning Convergence")
        ax1.grid(True, alpha=0.3)

        ax2.plot(history["iteration"], history["cumulative_time_ms"], marker="s", color="#d62728", lw=2)
        ax2.set_xlabel("Iteration")
        ax2.set_ylabel("Cumulative Compute (ms)")
        ax2.grid(True, alpha=0.3)
        plt.tight_layout()
        out_path = os.path.join(args.output_dir, "benchmark_active_learning.png")
        plt.savefig(out_path, dpi=200)
        plt.close()
        print(f"Saved benchmark convergence plot to {out_path}")

    print("\nBenchmark completed successfully.")


if __name__ == "__main__":
    main()
