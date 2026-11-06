"""Recompute the fixed-width history-state compression results.

Default: redraw bundled test results. Use --run-root runs to import evaluations.
Only NumPy/Matplotlib are required; paired statistics are shared with experiment 01.
"""

import argparse
import csv
import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent/"01_history_semantics"))
from metrics import NOISE_LEVELS, _aggregate, _pair

SEEDS = (42,43,44)
TOKENS = ("K64","K16","K4","K1")
CURVE = ("K1","K4","K16","K64")
ARMS = ("N","L",*TOKENS)
KIB = {"N":0,"L":72,"K64":128,"K16":32,"K4":8,"K1":2}
PAIRS = {**{f"N_vs_{a}":("N",a) for a in (*TOKENS,"L")},
         **{f"K64_vs_{a}":("K64",a) for a in TOKENS[1:]}}


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def reference_files(seed,q1_results,c64_results):
    return (q1_results/f"seed{seed}.npz",c64_results/f"seed{seed}.npz")


def import_runs(run_root,output,split,q1_results,c64_results):
    output.mkdir(parents=True,exist_ok=True)
    for seed in SEEDS:
        source=run_root/f"history_compression_seed{seed}"
        q1_path,_=reference_files(seed,q1_results,c64_results)
        with np.load(source/f"errors_step06000_{split}.npz",allow_pickle=False) as data, np.load(q1_path,allow_pickle=False) as old:
            if not np.array_equal(data["ids"],old["ids"]):raise ValueError(f"Q1 sample order differs: {seed}")
            values={k:data[k] for arm in TOKENS for k in (arm,arm+"_repeats")}
            values.update({k:old[k] for arm in ("N","L") for k in (arm,arm+"_repeats")})
            values.update({k:data[k] for k in ("ids","noise_levels","noise_seed","noise_repeats")})
            np.savez_compressed(output/f"seed{seed}.npz",**values)
    rows=read(run_root/"history_compression_seed42"/f"{split}_manifest.json")
    (output/f"{split}_manifest.json").write_text(json.dumps(
        [{k:r[k] for k in ("id","video","group")} for r in rows],indent=2),encoding="utf-8")


def load_results(directory,split,run_root,q1_results,c64_results):
    rows=read(directory/f"{split}_manifest.json")
    count,noise_seed=759,20042
    if len(rows)!=count:raise ValueError(f"Expected {count} {split} windows")
    runs=[]
    for seed in SEEDS:
        q1_path,c64_path=reference_files(seed,q1_results,c64_results)
        with np.load(directory/f"seed{seed}.npz",allow_pickle=False) as data:
            if (not np.array_equal(data["ids"],[r["id"] for r in rows])
                    or not np.array_equal(data["noise_levels"],NOISE_LEVELS)
                    or int(data["noise_seed"])!=noise_seed or int(data["noise_repeats"])!=4):
                raise ValueError(f"Misaligned evaluation: {seed}")
            errors={}
            for arm in ARMS:
                repeated=data[arm+"_repeats"]
                if (repeated.shape!=(4,count,5) or not np.isfinite(repeated).all()
                        or not np.array_equal(data[arm],repeated.mean(0))):
                    raise ValueError(f"Invalid repeated errors: {seed}/{arm}")
                errors[arm]=data[arm].copy()
            with np.load(q1_path,allow_pickle=False) as q1,np.load(c64_path,allow_pickle=False) as c64:
                if (not np.array_equal(data["ids"],q1["ids"]) or not np.array_equal(data["ids"],c64["ids"])):
                    raise ValueError(f"Reference sample order differs: {seed}")
                exact={"N":bool(np.array_equal(data["N_repeats"],q1["N_repeats"])),
                       "L":bool(np.array_equal(data["L_repeats"],q1["L_repeats"])),
                       "K64":bool(np.array_equal(data["K64_repeats"],c64["C_repeats"]))}
                if not all(exact.values()):raise ValueError(f"Fixed reference differs: {seed}/{exact}")
        arms,point,samples,counts=_aggregate(rows,errors,5000,2026)
        mse={a:float(point[i].mean()) for i,a in enumerate(arms)}
        denominator=mse["N"]-mse["K64"]
        retention={a:float((mse["N"]-mse[a])/denominator) if denominator>0 else None for a in TOKENS}
        runs.append({"seed":seed,"mse":mse,"retention":retention,"counts":counts,"reference_exact":exact,
                     "pairs":{k:_pair(arms,point,samples,a,b) for k,(a,b) in PAIRS.items()}})
    return runs


def describe(values):
    values=np.asarray(values,dtype=float)
    return {"mean":float(values.mean()),"min":float(values.min()),"max":float(values.max()),
            "sample_sd":float(values.std(ddof=1))}


def write_summary(runs,directory,split):
    records=[]
    for run in runs:
        for key,report in run["pairs"].items():
            p=report["primary"];lo,hi=p["ci95"]["relative_improvement"];a,b=PAIRS[key]
            records.append({"seed":run["seed"],"comparison":key,"state_kib":KIB[b],
                            "mse_reference":p["mse_a"],"mse_treatment":p["mse_b"],
                            "gain_percent":100*p["relative_improvement"],
                            "ci95_low_percent":100*lo,"ci95_high_percent":100*hi,
                            "gain_retention":run["retention"].get(b)})
    result={"split":split,"steps":6000,"seeds":list(SEEDS),"noise_repeats":4,
            "noise_seed":20042,"counts":runs[0]["counts"],
            "state_kib":KIB,"state_definition":"FP16 persistent state only; K* has 1024 dimensions per token. N has no state; L is the original A latent.",
            "token_model_parameters":14829968,"parameter_scope":"K64/K16/K4/K1 joint models; N/L are older background references.",
            "statistics":"Repeat mean -> window -> video -> source group -> class macro; paired source-group bootstrap within class, seed2026, 5000 draws. CIs are within-seed; three-seed mean/range/SD are descriptive.",
            "mse":{a:describe([r["mse"][a] for r in runs]) for a in ARMS},
            "gain_percent":{key:describe([100*r["pairs"][key]["primary"]["relative_improvement"] for r in runs]) for key in PAIRS},
            "gain_retention":{"definition":"R_K=(MSE_N-MSE_K)/(MSE_N-MSE_K64)",
                "scope":"Descriptive MSE-gain retention, not semantic information; no ratio CI, no clipping.",
                "per_seed":[{"seed":r["seed"],**r["retention"]} for r in runs],
                "across_seeds":{a:describe([r["retention"][a] for r in runs])
                                if all(r["retention"][a] is not None for r in runs) else None for a in TOKENS}},
            "reference_exact":[{"seed":r["seed"],"matches":r["reference_exact"]} for r in runs],"per_seed":records}
    (directory/"summary.json").write_text(json.dumps(result,indent=2),encoding="utf-8")
    with (directory/"summary.csv").open("w",newline="",encoding="utf-8") as stream:
        writer=csv.DictWriter(stream,fieldnames=list(records[0]));writer.writeheader();writer.writerows(records)
    return result


def plot(runs,directory,split):
    plt.rcParams.update({"font.size":10,"axes.titlesize":11,"legend.fontsize":9})
    fig,axes=plt.subplots(1,2,figsize=(12,4.8),constrained_layout=True)
    x=np.array([KIB[a] for a in CURVE])
    for run in runs:
        axes[0].plot(x,[run["mse"][a] for a in CURVE],color="#0072B2",alpha=.25,lw=1)
        axes[1].plot(x,[np.nan if run["retention"][a] is None else 100*run["retention"][a] for a in CURVE],"o-",alpha=.45,lw=1,label=f"seed {run['seed']}")
    axes[0].plot(x,[np.mean([r["mse"][a] for r in runs]) for a in CURVE],"o-",color="#0072B2",lw=2,label="Token-state mean")
    n=np.mean([r["mse"]["N"] for r in runs]);l=np.mean([r["mse"]["L"] for r in runs])
    axes[0].axhline(n,color="#607080",ls="--",lw=1.3,label="No history (0 KiB)")
    axes[0].plot(72,l,"D",color="black",label="Full history latent (72 KiB)")
    axes[0].set_xscale("log",base=2)
    axes[0].set(title=f"State cost versus overall {split} error",xlabel="Persistent FP16 state (KiB)",ylabel="Velocity MSE",xticks=[2,8,32,72,128],xticklabels=["2","8","32","72","128"])
    axes[0].legend(loc="best")
    axes[1].plot(x,[100*np.mean([r["retention"][a] for r in runs]) if all(r["retention"][a] is not None for r in runs) else np.nan for a in CURVE],"o-",color="black",lw=2,label="Seed mean")
    axes[1].axhline(100,color="grey",ls="--",lw=.8)
    axes[1].axhline(0,color="grey",lw=.6)
    axes[1].set_xscale("log",base=2)
    axes[1].set(title="MSE benefit retained relative to 64 tokens",xlabel="Persistent FP16 state (KiB)",ylabel="R (%) — descriptive, no ratio CI",xticks=x,xticklabels=["2\n1 token","8\n4 tokens","32\n16 tokens","128\n64 tokens"])
    axes[1].legend(loc="best")
    fig.suptitle("Task-driven history compression — 1024 dimensions/token, paired seeds 42/43/44, step 6000",fontsize=12)
    fig.savefig(directory/"comparison.png",dpi=180)
    plt.close(fig)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root",type=Path)
    parser.add_argument("--results-dir",type=Path)
    parser.add_argument("--split",choices=("test",),default="test")
    parser.add_argument("--q1-results",type=Path,default=HERE.parent/"01_history_semantics/results")
    parser.add_argument("--c64-results",type=Path,default=HERE.parent/"03_semantic_supervision/results")
    args=parser.parse_args()
    output=args.results_dir or (args.run_root/"history_compression_results_test" if args.run_root else HERE/"results")
    if args.run_root:import_runs(args.run_root,output,args.split,args.q1_results,args.c64_results)
    runs=load_results(output,args.split,args.run_root,args.q1_results,args.c64_results)
    result=write_summary(runs,output,args.split);plot(runs,output,args.split)
    print(json.dumps({"mse":result["mse"],"gain_retention":result["gain_retention"]},indent=2))


if __name__=="__main__":main()
