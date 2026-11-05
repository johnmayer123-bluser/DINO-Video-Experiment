"""Summarize paired FM-only C versus semantically supervised J.

Default: redraw bundled test results. Use --run-root runs to import a new run.
NumPy/Matplotlib only; shared paired statistics come from experiment 01.
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
sys.path.insert(0, str(HERE.parent/"01_history_semantics"))
from metrics import NOISE_LEVELS, _aggregate, _pair

ARMS = ("N", "P", "C", "J")
PAIRS = {"C_vs_J":("C","J"), "N_vs_C":("N","C"), "N_vs_J":("N","J"),
         "N_vs_P":("N","P"), "P_vs_C":("P","C"), "P_vs_J":("P","J")}
ROLES = {key:("primary" if key=="C_vs_J" else "secondary" if key in ("N_vs_C","N_vs_J") else "background") for key in PAIRS}
# Two-sided 95% Student-t critical values, df 1..9; no SciPy runtime dependency.
T975 = {1:12.706204736174605,2:4.302652729749449,3:3.1824463052837064,
        4:2.7764451051977845,5:2.570581835636318,6:2.4469118511449706,
        7:2.3646242515927813,8:2.3060041352041667,9:2.2621571627982}


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def import_runs(run_root, output, split, seeds):
    output.mkdir(parents=True,exist_ok=True)
    for seed in seeds:
        source = run_root/f"semantic_supervision_seed{seed}"
        with np.load(source/f"errors_step06000_{split}.npz",allow_pickle=False) as data:
            keys=[k for arm in ARMS if arm in data for k in (arm,arm+"_repeats")]
            keys += ["ids","noise_levels","noise_seed","noise_repeats"]
            np.savez_compressed(output/f"seed{seed}.npz",**{key:data[key] for key in keys})
    rows=read(run_root/f"semantic_supervision_seed{seeds[0]}"/f"{split}_manifest.json")
    (output/f"{split}_manifest.json").write_text(json.dumps(
        [{key:r[key] for key in ("id","video","group")} for r in rows],indent=2),encoding="utf-8")


def load_results(directory,split,references,seeds=tuple(range(42,52))):
    rows=read(directory/f"{split}_manifest.json")
    count,noise_seed=(759,20042) if split=="test" else (475,10042)
    if len(rows)!=count:
        raise ValueError(f"Expected {count} {split} windows")
    runs=[]
    for seed in seeds:
        with np.load(directory/f"seed{seed}.npz",allow_pickle=False) as data:
            if (not np.array_equal(data["ids"],[r["id"] for r in rows])
                    or not np.array_equal(data["noise_levels"],NOISE_LEVELS)
                    or int(data["noise_seed"])!=noise_seed or int(data["noise_repeats"])!=4):
                raise ValueError(f"Misaligned evaluation: seed {seed}")
            errors={}
            if "C" not in data or "J" not in data:
                raise ValueError(f"Missing paired C/J: seed {seed}")
            for arm in (a for a in ARMS if a in data):
                repeated=data[arm+"_repeats"]
                if (repeated.shape!=(4,count,5) or not np.isfinite(repeated).all()
                        or not np.array_equal(data[arm],repeated.mean(0))):
                    raise ValueError(f"Invalid repeated errors: {seed}/{arm}")
                errors[arm]=data[arm].copy()
            exact=None
            if split=="test" and "N" in errors and "P" in errors:
                with np.load(references/f"seed{seed}.npz",allow_pickle=False) as old:
                    exact={arm:bool(np.array_equal(data[arm+"_repeats"],old[arm+"_repeats"])) for arm in ("N","P")}
                    if not np.array_equal(data["ids"],old["ids"]) or not all(exact.values()):
                        raise ValueError(f"Fixed N/P reference differs: seed {seed}")
        arms,point,samples,counts=_aggregate(rows,errors,5000,2026)
        runs.append({"seed":seed,"mse":{arm:float(point[i].mean()) for i,arm in enumerate(arms)},
                     "pairs":{key:_pair(arms,point,samples,a,b) for key,(a,b) in PAIRS.items() if a in arms and b in arms},
                     "counts":counts,"reference_exact":exact})
    return runs


def describe(values):
    values=np.asarray(values,dtype=float)
    return {"n_seeds":len(values),"mean":float(values.mean()),"min":float(values.min()),"max":float(values.max()),
            "sample_sd":float(values.std(ddof=1)) if len(values)>1 else None}


def seed_uncertainty(runs):
    values=np.array([100*r["pairs"]["C_vs_J"]["primary"]["relative_improvement"] for r in runs])
    n=len(values);out=describe(values)
    half=T975[n-1]*out["sample_sd"]/np.sqrt(n) if n>1 else None
    out.update(ci95_seed_mean_percent=[out["mean"]-half,out["mean"]+half] if half is not None else None,
               wins=int((values>0).sum()),losses=int((values<0).sum()),ties=int((values==0).sum()),
               method="Paired seed-level Student-t interval for mean percentage improvement; fixed shared dataset, training randomness only. Approximate normal-mean interval, distinct from each seed's source-group bootstrap CI.")
    return out


def write_summary(runs,directory,split):
    records=[]
    for run in runs:
        for key,report in run["pairs"].items():
            p=report["primary"];lo,hi=p["ci95"]["relative_improvement"]
            records.append({"seed":run["seed"],"comparison":key,"role":ROLES[key],
                            "mse_reference":p["mse_a"],"mse_treatment":p["mse_b"],
                            "gain_percent":100*p["relative_improvement"],
                            "ci95_low_percent":100*lo,"ci95_high_percent":100*hi})
    result={"split":split,"steps":6000,"seeds":[r["seed"] for r in runs],"noise_repeats":4,
            "noise_seed":20042 if split=="test" else 10042,"counts":runs[0]["counts"],
            "conditions":{"C":"joint encoder+denoiser, FM only",
                          "J":"same architecture/initialization, FM + cosine alignment, lambda=1",
                          "N":"fixed Q1 no-history background","P":"fixed Q2 pretrained-q background"},
            "paired_design":"C/J: no pretraining; normalized 64x1024 condition, 10% dropout, zero history; FM backpropagates into encoder; shared data/noise/drop streams.",
            "primary":"C_vs_J overall velocity MSE","secondary":["N_vs_C","N_vs_J"],
            "background_only":["N_vs_P","P_vs_C","P_vs_J"],
            "statistics":"Noise mean -> window -> video -> source group -> class macro; paired within-class source-group bootstrap, seed2026, 5000 draws. Separate paired seed-level t interval; no pooling of shared videos across seeds.",
            "primary_seed_uncertainty":seed_uncertainty(runs),
            "background_seeds":[r["seed"] for r in runs if "N" in r["mse"]],
            "mse":{arm:describe([r["mse"][arm] for r in runs if arm in r["mse"]]) for arm in ARMS if any(arm in r["mse"] for r in runs)},
            "gain_percent":{key:describe([100*r["pairs"][key]["primary"]["relative_improvement"] for r in runs if key in r["pairs"]]) for key in PAIRS if any(key in r["pairs"] for r in runs)},
            "per_seed":records,"reference_exact":[{"seed":r["seed"],"matches":r["reference_exact"]} for r in runs]}
    (directory/"summary.json").write_text(json.dumps(result,indent=2),encoding="utf-8")
    with (directory/"summary.csv").open("w",newline="",encoding="utf-8") as stream:
        writer=csv.DictWriter(stream,fieldnames=list(records[0]));writer.writeheader();writer.writerows(records)
    return result


def plot(runs,directory,split):
    plt.rcParams.update({"font.size":10,"axes.titlesize":12,"legend.fontsize":9})
    fig,axes=plt.subplots(1,2,figsize=(11.5,6.2),constrained_layout=True)
    means=[np.mean([r["mse"][arm] for r in runs]) for arm in ("C","J")]
    axes[0].bar(range(2),means,color=["#D55E00","#9B59B6"],width=.6,alpha=.85)
    for i,value in enumerate(means):axes[0].text(i,value+max(means)*.008,f"{value:.4f}",ha="center")
    axes[0].set(title=f"Overall {split} error: same {len(runs)} seeds",ylabel="Velocity MSE (lower is better)",xticks=range(2),
                xticklabels=["C\nFM only","J\nFM + alignment"],ylim=(0,max(means)*1.12))
    for i,run in enumerate(runs):
        p=run["pairs"]["C_vs_J"]["primary"];lo,hi=100*np.array(p["ci95"]["relative_improvement"])
        gain=100*p["relative_improvement"];color="#0072B2" if gain>=0 else "#D55E00"
        axes[1].hlines(i,lo,hi,color=color)
        axes[1].plot(gain,i,"o",color=color)
    seed_stats=seed_uncertainty(runs);interval=seed_stats["ci95_seed_mean_percent"]
    if interval is not None:axes[1].hlines(len(runs),*interval,color="black",lw=2)
    axes[1].plot(seed_stats["mean"],len(runs),"D",color="black")
    axes[1].axvline(0,color="grey",lw=.8)
    axes[1].set(title="J over C — two distinct 95% intervals",xlabel="MSE reduction (%)",
                yticks=range(len(runs)+1),yticklabels=[f"Seed {r['seed']}" for r in runs]+["Seed mean\n(training t CI)"])
    axes[1].invert_yaxis()
    axes[1].set_xlabel("MSE reduction (%)\nSeed rows: source-group CI; mean row: seed t CI",fontsize=9)
    fig.suptitle("Semantic supervision — paired C/J, final step 6000",fontsize=12)
    fig.savefig(directory/"comparison.png",dpi=180)
    plt.close(fig)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root",type=Path)
    parser.add_argument("--results-dir",type=Path)
    parser.add_argument("--split",choices=("val","test"),default="test")
    parser.add_argument("--seeds",type=int,nargs="+",default=list(range(42,52)),choices=range(42,52))
    parser.add_argument("--references",type=Path,default=HERE.parent/"02_predicted_semantics/results")
    args=parser.parse_args()
    if len(set(args.seeds))!=len(args.seeds):parser.error("Each seed must appear once")
    seeds=sorted(args.seeds)
    output=args.results_dir or (args.run_root/f"semantic_supervision_results_{args.split}" if args.run_root else
                               HERE/"results" if args.split=="test" else HERE/"results/val")
    if args.run_root:import_runs(args.run_root,output,args.split,seeds)
    runs=load_results(output,args.split,args.references,seeds)
    result=write_summary(runs,output,args.split);plot(runs,output,args.split)
    print(json.dumps(result["gain_percent"],indent=2))


if __name__=="__main__":main()
