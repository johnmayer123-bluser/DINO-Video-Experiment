"""Analyze the fixed-16-token width sweep using raw paired test errors.

Default: redraw bundled results. New evaluations: --run-root runs.
NumPy/Matplotlib only; shared statistics come from experiment 01.
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

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parents[1]/"01_history_semantics"))
from metrics import NOISE_LEVELS,_aggregate,_pair

SEEDS=(42,43,44)
WIDTHS=("D1024","D512","D256","D128")
CURVE=tuple(reversed(WIDTHS))
REFERENCES=("N","L","K64","K16","K4","K1")
KIB={"D1024":32,"D512":16,"D256":8,"D128":4}
PARAMETERS={"D1024":14829968,"D512":14567312,"D256":14435984,"D128":14370320}
PAIRS={**{f"N_vs_{a}":("N",a) for a in WIDTHS},
       **{f"D1024_vs_{a}":("D1024",a) for a in WIDTHS[1:]},
       "K4_vs_D256":("K4","D256")}


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def import_runs(run_root,output):
    output.mkdir(parents=True,exist_ok=True)
    for seed in SEEDS:
        source=run_root/f"history_width_seed{seed}"
        with np.load(source/"errors_step06000_test.npz",allow_pickle=False) as data:
            keys=[k for arm in WIDTHS for k in (arm,arm+"_repeats")]
            keys += ["ids","noise_levels","noise_seed","noise_repeats"]
            np.savez_compressed(output/f"seed{seed}.npz",**{key:data[key] for key in keys})
    rows=read(run_root/"history_width_seed42/test_manifest.json")
    (output/"test_manifest.json").write_text(json.dumps(
        [{k:r[k] for k in ("id","video","group")} for r in rows],indent=2),encoding="utf-8")


def load_results(directory,references):
    rows=read(directory/"test_manifest.json")
    if len(rows)!=759:raise ValueError("Expected 759 fixed test windows")
    runs=[]
    for seed in SEEDS:
        with np.load(directory/f"seed{seed}.npz",allow_pickle=False) as data,np.load(references/f"seed{seed}.npz",allow_pickle=False) as old:
            for source in (data,old):
                if (not np.array_equal(source["ids"],[r["id"] for r in rows])
                        or not np.array_equal(source["noise_levels"],NOISE_LEVELS)
                        or int(source["noise_seed"])!=20042 or int(source["noise_repeats"])!=4):
                    raise ValueError(f"Misaligned test errors: seed {seed}")
            errors={}
            for arm in (*REFERENCES,*WIDTHS):
                source=old if arm in REFERENCES else data
                repeated=source[arm+"_repeats"]
                if (repeated.shape!=(4,759,5) or not np.isfinite(repeated).all()
                        or not np.array_equal(source[arm],repeated.mean(0))):
                    raise ValueError(f"Invalid repeated errors: {seed}/{arm}")
                errors[arm]=source[arm].copy()
            if not np.array_equal(data["D1024_repeats"],old["K16_repeats"]):
                raise ValueError(f"D1024 differs from the fixed Q4 K16 baseline: {seed}")
        arms,point,samples,counts=_aggregate(rows,errors,5000,2026)
        mse={arm:float(point[i].mean()) for i,arm in enumerate(arms)}
        retained={}
        for name,baseline in (("local","K16"),("original","K64")):
            denominator=mse["N"]-mse[baseline]
            retained[name]={arm:float((mse["N"]-mse[arm])/denominator) if denominator>0 else None for arm in WIDTHS}
        runs.append({"seed":seed,"mse":mse,"retention":retained,"counts":counts,
                     "pairs":{key:_pair(arms,point,samples,a,b) for key,(a,b) in PAIRS.items()}})
    return runs


def describe(values):
    values=np.asarray(values,dtype=float)
    return {"mean":float(values.mean()),"min":float(values.min()),"max":float(values.max()),
            "sample_sd":float(values.std(ddof=1))}


def write_summary(runs,directory):
    records=[]
    for run in runs:
        for key,report in run["pairs"].items():
            p=report["primary"];lo,hi=p["ci95"]["relative_improvement"];_,arm=PAIRS[key]
            records.append({"seed":run["seed"],"comparison":key,
                            "role":"exploratory_equal_bytes" if key=="K4_vs_D256" else "width_sweep",
                            "state_kib":KIB[arm],"mse_reference":p["mse_a"],"mse_treatment":p["mse_b"],
                            "gain_percent":100*p["relative_improvement"],
                            "ci95_low_percent":100*lo,"ci95_high_percent":100*hi,
                            "retention_local":run["retention"]["local"][arm],
                            "retention_original":run["retention"]["original"][arm]})
    result={"split":"test","steps":6000,"seeds":list(SEEDS),"tokens":16,"state_kib":KIB,
            "model_parameters":PARAMETERS,
            "parameter_note":"Each removed width dimension removes 513 parameters from the encoder output head and attention context projection.",
            "noise_seed":20042,"noise_repeats":4,"counts":runs[0]["counts"],
            "statistics":"Repeat mean -> window -> video -> source group -> class macro; paired within-class source-group bootstrap, 2026/5000. Per-seed CIs; three-seed mean/range/SD are descriptive.",
            "mse":{arm:describe([r["mse"][arm] for r in runs]) for arm in (*REFERENCES,*WIDTHS)},
            "gain_percent":{key:describe([100*r["pairs"][key]["primary"]["relative_improvement"] for r in runs]) for key in PAIRS},
            "retention":{"definitions":{"local":"(N-D)/(N-Q4K16)","original":"(N-D)/(N-Q3C64)"},
                "scope":"MSE-gain retention only, not semantic information; ratios are computed within each seed before averaging. Descriptive points, no ratio CI or clipping.",
                "per_seed":[{"seed":r["seed"],**r["retention"]} for r in runs],
                "across_seeds":{name:{arm:describe([r["retention"][name][arm] for r in runs])
                                     if all(r["retention"][name][arm] is not None for r in runs) else None
                                     for arm in WIDTHS} for name in ("local","original")}},
            "baseline_check":"All D1024 repeated errors exactly equal Q4 K16 for seeds 42/43/44.",
            "equal_byte_comparison":"K4x1024 versus K16x256 (both 8 KiB) is exploratory, not an additional fixed-16 width contrast.",
            "per_seed":records}
    (directory/"summary.json").write_text(json.dumps(result,indent=2),encoding="utf-8")
    with (directory/"summary.csv").open("w",newline="",encoding="utf-8") as stream:
        writer=csv.DictWriter(stream,fieldnames=list(records[0]));writer.writeheader();writer.writerows(records)
    return result


def plot(runs,directory):
    plt.rcParams.update({"font.size":9,"axes.titlesize":10,"legend.fontsize":8})
    fig,axes=plt.subplots(1,3,figsize=(16,4.8),constrained_layout=True)
    x=np.array([KIB[arm] for arm in CURVE])
    for run in runs:
        axes[0].plot(x,[run["mse"][arm] for arm in CURVE],color="#0072B2",alpha=.25,lw=1)
    axes[0].plot(x,[np.mean([r["mse"][arm] for r in runs]) for arm in CURVE],"o-",color="#0072B2",lw=2,label="16-token width mean")
    axes[0].axhline(np.mean([r["mse"]["N"] for r in runs]),color="#607080",ls="--",lw=1,label="No history (0 KiB)")
    for kib,arm,marker,label in ((72,"L","D","Full history latent (72 KiB)"),(128,"K64","^","64 tokens (128 KiB)"),(8,"K4","x","4 tokens × 1024 dims (exploratory)")):
        axes[0].plot(kib,np.mean([r["mse"][arm] for r in runs]),marker,color="black",label=label)
    axes[0].set_xscale("log",base=2)
    axes[0].set(title="State cost versus overall test error",xlabel="Persistent FP16 state (KiB)",ylabel="Velocity MSE",xticks=[4,8,16,32,72,128],xticklabels=["4","8","16","32","72","128"])
    axes[0].legend(loc="best")
    for axis,name,title in ((axes[1],"local","Relative to 16 tokens × 1024 dims"),(axes[2],"original","Relative to 64 tokens × 1024 dims")):
        for run in runs:
            axis.plot(x,[np.nan if run["retention"][name][arm] is None else 100*run["retention"][name][arm] for arm in CURVE],"o-",alpha=.45,lw=1,label=f"seed {run['seed']}")
        mean=[100*np.mean([r["retention"][name][arm] for r in runs]) if all(r["retention"][name][arm] is not None for r in runs) else np.nan for arm in CURVE]
        axis.plot(x,mean,"o-",color="black",lw=2,label="Seed mean")
        axis.axhline(100,color="grey",ls="--",lw=.8);axis.axhline(0,color="grey",lw=.6)
        axis.set_xscale("log",base=2)
        axis.set(title=title,xlabel="FP16 state KiB (dimensions/token)",ylabel="MSE-gain retention (%) — descriptive",xticks=x,xticklabels=["4\n128","8\n256","16\n512","32\n1024"])
        axis.legend(loc="best")
    fig.suptitle("History compression — fixed 16 tokens, paired seeds 42/43/44, step 6000",fontsize=12)
    fig.savefig(directory/"comparison.png",dpi=180)
    plt.close(fig)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root",type=Path)
    parser.add_argument("--results-dir",type=Path)
    parser.add_argument("--split",choices=("test",),default="test")
    parser.add_argument("--references",type=Path,default=HERE.parent/"results")
    args=parser.parse_args()
    output=args.results_dir or (args.run_root/"history_width_results_test" if args.run_root else HERE/"results")
    if args.run_root:import_runs(args.run_root,output)
    runs=load_results(output,args.references);result=write_summary(runs,output);plot(runs,output)
    print(json.dumps({"mse":result["mse"],"retention":result["retention"]},indent=2))


if __name__=="__main__":main()
