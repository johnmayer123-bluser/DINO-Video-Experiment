"""Summarize frozen-q replacement and denoiser adaptation from paired raw errors.

Default: redraw bundled test results. New runs: --run-root runs --results-dir PATH.
Only NumPy/Matplotlib are required; statistics are shared with experiment 01.
"""

import argparse
import csv
import json
from pathlib import Path
import shutil
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent/"01_history_semantics"))
from metrics import NOISE_LEVELS, _aggregate, _pair

ARMS = ("N", "T", "D", "P")
PAIRS = {"N_vs_T":("N","T"), "N_vs_D":("N","D"), "N_vs_P":("N","P"),
         "T_vs_D":("T","D"), "T_vs_P":("T","P"), "D_vs_P":("D","P")}
COLORS = ("#607080", "#9B59B6", "#D55E00", "#0072B2")


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def import_runs(run_root, output, split):
    output.mkdir(parents=True, exist_ok=True)
    for seed in (42,43,44):
        source = run_root/f"predicted_semantics_seed{seed}"
        with np.load(source/f"errors_step06000_{split}.npz", allow_pickle=False) as data:
            keys = [k for arm in ARMS for k in (arm,arm+"_repeats")]
            keys += ["ids","noise_levels","noise_seed","noise_repeats"]
            np.savez_compressed(output/f"seed{seed}.npz", **{k:data[k] for k in keys})
    rows = read(run_root/"predicted_semantics_seed42"/f"{split}_manifest.json")
    (output/f"{split}_manifest.json").write_text(json.dumps(
        [{k:r[k] for k in ("id","video","group")} for r in rows], indent=2),encoding="utf-8")


def load_results(directory, split, references):
    rows = read(directory/f"{split}_manifest.json")
    count, noise_seed = (759,20042) if split == "test" else (475,10042)
    if len(rows) != count:
        raise ValueError(f"Expected {count} {split} windows")
    runs = []
    for seed in (42,43,44):
        with np.load(directory/f"seed{seed}.npz",allow_pickle=False) as data:
            if (not np.array_equal(data["ids"],[r["id"] for r in rows])
                    or not np.array_equal(data["noise_levels"],NOISE_LEVELS)
                    or int(data["noise_seed"]) != noise_seed or int(data["noise_repeats"]) != 4):
                raise ValueError(f"Misaligned evaluation: seed {seed}")
            errors = {}
            for arm in ARMS:
                repeated = data[arm+"_repeats"]
                if (repeated.shape != (4,count,5) or not np.isfinite(repeated).all()
                        or not np.array_equal(data[arm],repeated.mean(0))):
                    raise ValueError(f"Invalid repeated errors: {seed}/{arm}")
                errors[arm] = data[arm].copy()
            reference_check = None
            if split == "test":
                with np.load(references/f"seed{seed}.npz",allow_pickle=False) as old:
                    reference_check = {"N":bool(np.array_equal(data["N_repeats"],old["N_repeats"])),
                                       "T":bool(np.array_equal(data["T_repeats"],old["SD_repeats"]))}
                    if not np.array_equal(data["ids"],old["ids"]) or not all(reference_check.values()):
                        raise ValueError(f"Q1 N/SD reference differs: seed {seed}")
        arms, point, samples, counts = _aggregate(rows,errors,5000,2026)
        mse = {arm:float(point[i].mean()) for i,arm in enumerate(arms)}
        retained = {arm:float((mse["N"]-mse[arm])/(mse["N"]-mse["T"]))
                    if mse["N"] > mse["T"] else None for arm in ("D","P")}
        runs.append({"seed":seed,"mse":mse,"counts":counts,"teacher_benefit_retained":retained,
                     "pairs":{k:_pair(arms,point,samples,a,b) for k,(a,b) in PAIRS.items()},
                     "q1_reference_exact":reference_check})
    return runs


def describe(values):
    values = np.asarray(values,dtype=float)
    return {"mean":float(values.mean()),"min":float(values.min()),"max":float(values.max()),
            "sample_sd":float(values.std(ddof=1))}


def write_summary(runs, directory, split):
    records = []
    for run in runs:
        for comparison, report in run["pairs"].items():
            p = report["primary"]
            lo,hi = p["ci95"]["relative_improvement"]
            records.append({"seed":run["seed"],"comparison":comparison,"mse_reference":p["mse_a"],
                            "mse_treatment":p["mse_b"],"gain_percent":100*p["relative_improvement"],
                            "ci95_low_percent":100*lo,"ci95_high_percent":100*hi})
    teacher_losses = {key: [{"seed":r["seed"],"loss_percent":-r["gain_percent"],
                            "ci95_percent":[-r["ci95_high_percent"],-r["ci95_low_percent"]]}
                           for r in records if r["comparison"] == key] for key in ("T_vs_D","T_vs_P")}
    result = {"split":split,"steps":6000,"seeds":[42,43,44],"q_seed":42,"q_step":2800,
              "noise_seed":20042 if split == "test" else 10042,"noise_repeats":4,"counts":runs[0]["counts"],
              "conditions":{"N":"no history","T":"frozen teacher-conditioned SD",
                            "D":"T weights, predicted q(z_A) condition only",
                            "P":"fresh denoiser trained on frozen q(z_A), 10% conditional dropout, zero history"},
              "statistics":"Window noise mean -> video -> source group -> class macro; paired within-class source-group bootstrap, 2026/5000. CIs are per-seed; cross-seed summaries are descriptive.",
              "mse":{arm:describe([r["mse"][arm] for r in runs]) for arm in ARMS},
              "gain_percent":{key:describe([100*r["pairs"][key]["primary"]["relative_improvement"] for r in runs]) for key in PAIRS},
              "teacher_relative_loss":teacher_losses,
              "teacher_benefit_retained":{"definition":"R_X=(MSE_N-MSE_X)/(MSE_N-MSE_T)",
                "scope":"Descriptive point ratios only, no ratio CI; not clipped to [0,1]. Gain CIs remain the inferential results.",
                "per_seed":[{"seed":r["seed"],**r["teacher_benefit_retained"]} for r in runs],
                "across_seeds":{arm:describe([r["teacher_benefit_retained"][arm] for r in runs])
                                if all(r["teacher_benefit_retained"][arm] is not None for r in runs) else None for arm in ("D","P")}},
              "q1_reference_exact":[{"seed":r["seed"],"matches":r["q1_reference_exact"]} for r in runs],
              "per_seed":records}
    (directory/"summary.json").write_text(json.dumps(result,indent=2),encoding="utf-8")
    with (directory/"summary.csv").open("w",newline="",encoding="utf-8") as stream:
        writer=csv.DictWriter(stream,fieldnames=list(records[0]));writer.writeheader();writer.writerows(records)
    return result


def plot(runs,directory,split):
    plt.rcParams.update({"font.size":10,"axes.titlesize":11,"legend.fontsize":9})
    fig,axes=plt.subplots(1,3,figsize=(15,4.6),constrained_layout=True)
    means=[np.mean([r["mse"][arm] for r in runs]) for arm in ARMS]
    axes[0].bar(range(4),means,color=COLORS,width=.6,alpha=.8)
    span=max(means)*.02
    for i,value in enumerate(means):axes[0].text(i,value+span/3,f"{value:.4f}",ha="center")
    axes[0].set(title=f"Overall {split} error",ylabel="Velocity MSE",xticks=range(4),
                xticklabels=["N\nNone","T\nTeacher","D\nDirect swap","P\nAdapted"],ylim=(0,max(means)*1.12))
    for j,run in enumerate(runs):
        p=[run["pairs"][k]["primary"] for k in ("N_vs_T","N_vs_D","N_vs_P")]
        x=np.arange(3)+(j-1)*.12
        axes[1].vlines(x,[100*v["ci95"]["relative_improvement"][0] for v in p],
                       [100*v["ci95"]["relative_improvement"][1] for v in p],color=f"C{j}")
        axes[1].plot(x,[100*v["relative_improvement"] for v in p],"o",color=f"C{j}",label=f"seed {run['seed']}")
        ratios=[run["teacher_benefit_retained"][a] for a in ("D","P")]
        axes[2].plot(np.arange(2)+(j-1)*.12,[np.nan if r is None else 100*r for r in ratios],"o",color=f"C{j}",label=f"seed {run['seed']}")
    axes[1].axhline(0,color="black",lw=.8)
    axes[1].set(title="Benefit over no-history N",ylabel="MSE reduction (%)",xticks=range(3),xticklabels=["T","D","P"])
    axes[1].legend(title="95% source-group CI",loc="lower right")
    axes[2].axhline(100,color="grey",ls="--",lw=.9)
    axes[2].axhline(0,color="black",lw=.8)
    axes[2].set(title="Teacher benefit retained",ylabel="R (%) — descriptive, no ratio CI",xticks=[0,1],xticklabels=["D: direct swap","P: adapted"])
    axes[2].legend(loc="best")
    fig.suptitle("Predicted semantics — frozen q, three denoiser seeds, final step 6000",fontsize=13)
    fig.savefig(directory/"comparison.png",dpi=180)
    plt.close(fig)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root",type=Path)
    parser.add_argument("--results-dir",type=Path)
    parser.add_argument("--split",choices=("val","test"),default="test")
    parser.add_argument("--references",type=Path,default=HERE.parent/"01_history_semantics/results")
    args=parser.parse_args()
    output=args.results_dir or (args.run_root/f"predicted_semantics_results_{args.split}" if args.run_root else
                               HERE/"results" if args.split=="test" else HERE/"results/val")
    if args.run_root:import_runs(args.run_root,output,args.split)
    runs=load_results(output,args.split,args.references)
    result=write_summary(runs,output,args.split)
    plot(runs,output,args.split)
    print(json.dumps({"gain_percent":result["gain_percent"],"teacher_benefit_retained":result["teacher_benefit_retained"]},indent=2))


if __name__=="__main__":main()
