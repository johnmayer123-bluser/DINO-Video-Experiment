"""Recompute the compact experiment summary and main figure from raw test errors.

Run: python experiments/01_history_semantics/plot_results.py
Requires only NumPy and Matplotlib; no models, cache, or GPU are loaded.
"""

import argparse
import csv
import json
from pathlib import Path
import shutil

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from metrics import NOISE_LEVELS, summarize

ARMS = ("N", "S", "SD", "L")
COLORS = ("#607080", "#D55E00", "#9B59B6", "#0072B2")


def describe(values):
    values = np.asarray(values, dtype=float)
    return {"mean":float(values.mean()), "min":float(values.min()), "max":float(values.max()),
            "sample_sd":float(values.std(ddof=1))}


def load_results(directory):
    rows = json.loads((directory/"test_manifest.json").read_text(encoding="utf-8"))
    runs = []
    for seed in (42,43,44):
        with np.load(directory/f"seed{seed}.npz", allow_pickle=False) as data:
            if (not np.array_equal(data["ids"], [r["id"] for r in rows])
                    or not np.array_equal(data["noise_levels"], NOISE_LEVELS)
                    or int(data["noise_seed"]) != 20042 or int(data["noise_repeats"]) != 4):
                raise ValueError(f"Misaligned test errors for seed {seed}")
            errors = {}
            for arm in ARMS:
                repeated = data[arm+"_repeats"]
                if (repeated.shape != (4,len(rows),5) or not np.isfinite(repeated).all()
                        or not np.array_equal(data[arm], repeated.mean(0))):
                    raise ValueError(f"Invalid repeated errors: seed {seed}, {arm}")
                errors[arm] = repeated.mean(0)
        runs.append({"seed":seed, **summarize(rows, errors)})
    return runs


def write_summary(runs, directory):
    records = []
    for run in runs:
        for comparison, report in run["pairs"].items():
            point = report["primary"]
            lo, hi = point["ci95"]["relative_improvement"]
            records.append({"seed":run["seed"], "comparison":comparison, "mse_reference":point["mse_a"],
                            "mse_treatment":point["mse_b"], "improvement_percent":100*point["relative_improvement"],
                            "ci95_low_percent":100*lo, "ci95_high_percent":100*hi})
    result = {"split":"test", "steps":6000, "seeds":[42,43,44], "noise_seed":20042,"noise_repeats":4,
              "noise_levels":list(NOISE_LEVELS), "counts":runs[0]["counts"],
              "primary":"SD versus N, all five noise levels", "secondary":"SD versus S, all five noise levels",
              "statistics":"Repeat mean -> window -> video -> source group -> class macro; paired source-group bootstrap within class, seed 2026, 5000 draws.",
              "interval_scope":"Per-seed CIs measure test source-group sampling; cross-seed mean/range/sample SD are descriptive, with no pooled CI.",
              "test_status":"Follow-up evaluation: this test set was already examined in earlier Q1.",
              "test_mse":{arm:describe([r["mse"][arm] for r in runs]) for arm in ARMS},
              "test_improvement_percent":{key:describe([100*r["pairs"][key]["primary"]["relative_improvement"] for r in runs]) for key in runs[0]["pairs"]},
              "per_seed":records}
    validation = directory/"validation_overall.json"
    if validation.exists():
        result["validation_overall"] = json.loads(validation.read_text(encoding="utf-8"))
    (directory/"summary.json").write_text(json.dumps(result,indent=2),encoding="utf-8")
    with (directory/"summary.csv").open("w",newline="",encoding="utf-8") as stream:
        writer=csv.DictWriter(stream,fieldnames=list(records[0]));writer.writeheader();writer.writerows(records)
    return result


def plot(runs, directory):
    plt.rcParams.update({"font.size":10,"axes.titlesize":12,"legend.fontsize":9})
    fig, axes = plt.subplots(1,2,figsize=(11,4.5),constrained_layout=True)
    means=[np.mean([r["mse"][arm] for r in runs]) for arm in ARMS]
    axes[0].bar(range(4), means, color=COLORS, width=.6, alpha=.8)
    for i, value in enumerate(means):
        axes[0].text(i,value+.003,f"{value:.4f}",ha="center",fontsize=10)
    for j, run in enumerate(runs):
        axes[0].scatter(np.arange(4)+(j-1)*.10,[run["mse"][arm] for arm in ARMS],s=19,
                        facecolors="white",edgecolors="black",linewidths=.7,zorder=3)
    axes[0].set(title="Overall test error",ylabel="Velocity MSE (lower is better)",xticks=range(4),
                xticklabels=["No history","Teacher\nsemantics","Teacher +\ndropout","Full history\nlatent"],ylim=(0,.52))
    for j, run in enumerate(runs):
        points=[run["pairs"][p]["primary"] for p in ("N_vs_SD","S_vs_SD")]
        x=np.arange(2)+(j-1)*.12
        axes[1].vlines(x,[100*p["ci95"]["relative_improvement"][0] for p in points],
                       [100*p["ci95"]["relative_improvement"][1] for p in points],color=f"C{j}")
        axes[1].plot(x,[100*p["relative_improvement"] for p in points],"o",color=f"C{j}",label=f"seed {run['seed']}")
    axes[1].axhline(0,color="black",lw=.8)
    axes[1].set(title="Teacher + dropout: test benefit",ylabel="MSE reduction (%)",xticks=[0,1],
                xticklabels=["Teacher + dropout\nvs no history","Teacher + dropout\nvs teacher semantics"])
    axes[1].legend(title="95% source-group CI",loc="lower right")
    fig.suptitle("History semantics — final step 6000, three training seeds",fontsize=13)
    fig.savefig(directory/"comparison.png",dpi=180)
    plt.close(fig)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, help="Read new evaluate.py outputs under history_semantics_seed{seed}")
    parser.add_argument("--results-dir", "--results", dest="results", type=Path,
                        help="Output directory; defaults to bundled results, or RUN_ROOT/history_semantics_results")
    args=parser.parse_args()
    directory = args.results or (args.run_root/"history_semantics_results" if args.run_root else
                                 Path(__file__).resolve().parent/"results")
    if args.run_root:
        directory.mkdir(parents=True, exist_ok=True)
        for seed in (42,43,44):
            source = args.run_root/f"history_semantics_seed{seed}"
            shutil.copyfile(source/"errors_step06000_test.npz", directory/f"seed{seed}.npz")
        shutil.copyfile(args.run_root/"history_semantics_seed42"/"test_manifest.json", directory/"test_manifest.json")
    runs=load_results(directory)
    result=write_summary(runs,directory)
    plot(runs,directory)
    print(json.dumps(result["test_improvement_percent"],indent=2))


if __name__=="__main__":main()
