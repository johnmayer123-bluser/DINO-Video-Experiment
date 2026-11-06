"""Plot the paired denoiser A/B validation result."""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--step", type=int, default=3000)
    args = parser.parse_args()

    run_dir = args.run_dir.resolve()
    summary = read_json(run_dir / f"summary_step{args.step:05d}.json")
    config = read_json(run_dir / "config.json")
    records = [json.loads(line) for line in
               (run_dir / "metrics.jsonl").read_text(encoding="utf-8").splitlines()
               if line.strip()]
    # A resumed run can contain repeated evaluations; retain the latest record.
    by_step = {int(record["step"]): record for record in records
               if int(record["step"]) <= args.step}
    curve = [by_step[step] for step in sorted(by_step)]
    steps = np.asarray([record["step"] for record in curve])
    mse_a = np.asarray([record["mse_a"] for record in curve])
    mse_b = np.asarray([record["mse_b"] for record in curve])

    levels = np.asarray(summary["A_vs_B"]["noise_levels"], dtype=float)
    slices = summary["A_vs_B"]["slices"]
    improvements, lower, upper = [], [], []
    for level in levels:
        result = slices[f"t_{level:.1f}"]
        value = 100 * result["relative_improvement"]
        low, high = (100 * item for item in result["ci95"]["relative_improvement"])
        improvements.append(value)
        lower.append(max(0.0, value - low))
        upper.append(max(0.0, high - value))

    plt.rcParams.update({
        "font.size": 10,
        "axes.titlesize": 11,
        "axes.labelsize": 10,
        "legend.fontsize": 9,
    })
    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.3), constrained_layout=True)

    axes[0].plot(steps, mse_a, marker="o", markersize=3.5, linewidth=1.8,
                 color="#4C78A8", label="Latent only")
    axes[0].plot(steps, mse_b, marker="o", markersize=3.5, linewidth=1.8,
                 color="#E45756", label="Latent + prediction")
    axes[0].set_xlabel("Optimizer updates")
    axes[0].set_ylabel("Validation velocity MSE\n(class-macro mean over five t values)")
    axes[0].set_title("Equal-budget validation curves")
    positive = np.concatenate((mse_a[mse_a > 0], mse_b[mse_b > 0]))
    if len(positive) and positive.max() / positive.min() > 5:
        axes[0].set_yscale("log")
        axes[0].set_ylabel(axes[0].get_ylabel() + " [log scale]")
    axes[0].grid(True, alpha=0.25)
    axes[0].legend(frameon=False)

    axes[1].errorbar(levels, improvements, yerr=np.asarray([lower, upper]),
                     fmt="o-", markersize=5, linewidth=1.8, capsize=4,
                     color="#59A14F", ecolor="#3F7F39")
    axes[1].axhline(0, color="black", linewidth=1, alpha=0.7)
    axes[1].set_xticks(levels)
    axes[1].set_xlabel("Flow time t  (t = 0: pure noise; t = 1: clean latent)")
    axes[1].set_ylabel("MSE reduction from predicted semantics (%)")
    axes[1].set_title("Paired effect by noise level (95% CI)")
    axes[1].grid(True, axis="y", alpha=0.25)

    seed = config.get("seed", "unknown")
    fig.suptitle(
        f"Latent history with predicted semantics — seed {seed}, fixed step {args.step}\n"
        "Fixed validation noise; paired source-group bootstrap within class",
        fontsize=12,
    )
    for extension in ("png", "svg"):
        output = run_dir / f"ab_result_step{args.step:05d}.{extension}"
        fig.savefig(output, dpi=220 if extension == "png" else None,
                    bbox_inches="tight", facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    main()
