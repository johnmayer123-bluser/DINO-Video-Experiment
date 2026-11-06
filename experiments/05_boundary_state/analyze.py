"""Paired whole-chunk and temporal-region analysis of boundary/state conditions."""
import argparse
import csv
import json
from pathlib import Path
import sys
import shutil

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent/'01_history_semantics'))
from metrics import NOISE_LEVELS,_aggregate,_pair

ARMS=('S8','B18','H26','S26')
REGIONS=('overall','first','later')
PAIRS={'S26_vs_H26':('S26','H26'),'S8_vs_H26':('S8','H26'),
       'B18_vs_H26':('B18','H26'),'S8_vs_B18':('S8','B18'),'S8_vs_S26':('S8','S26')}
SEEDS=(42,43,44)
KIB={'S8':8,'B18':18,'H26':26,'S26':26}


def read(path):return json.loads(path.read_text(encoding='utf-8'))


def describe(values):
    x=np.asarray(values,dtype=float)
    return {'mean':float(x.mean()),'sample_sd':float(x.std(ddof=1)),'min':float(x.min()),'max':float(x.max())}


def import_runs(run_root,out,split):
    out.mkdir(parents=True,exist_ok=True)
    for seed in SEEDS:
        shutil.copy2(run_root/f'boundary_state_seed{seed}/errors_step06000_{split}.npz',out/f'seed{seed}.npz')
    rows=read(run_root/f'boundary_state_seed42/{split}_manifest.json')
    (out/f'{split}_manifest.json').write_text(json.dumps([{k:r[k] for k in ('id','video','group')} for r in rows],indent=2),encoding='utf-8')


def analyze(out,split):
    rows=read(out/f'{split}_manifest.json');runs=[];records=[]
    for seed in SEEDS:
        with np.load(out/f'seed{seed}.npz',allow_pickle=False) as data:
            assert np.array_equal(data['ids'],[r['id'] for r in rows])
            assert np.array_equal(data['noise_levels'],NOISE_LEVELS)
            assert int(data['noise_seed'])==(20042 if split=='test' else 10042)
            assert int(data['noise_repeats'])==4
            errors={r:{} for r in REGIONS}
            for arm in ARMS:
                repeated=data[arm+'_repeats'];frames=data[arm+'_by_frame_repeats']
                assert repeated.shape==(4,len(rows),5) and frames.shape==(4,len(rows),5,4)
                assert np.isfinite(repeated).all() and np.isfinite(frames).all()
                assert np.array_equal(data[arm],repeated.mean(0))
                assert np.allclose(repeated,frames.mean(-1),rtol=1e-6,atol=1e-7)
                errors['overall'][arm]=data[arm].copy()
                errors['first'][arm]=frames.mean(0)[...,0]
                errors['later'][arm]=frames.mean(0)[...,1:].mean(-1)
            reference=None
            if split=='test':
                old=HERE.parent/'04_history_compression/width_sweep/results'/f'seed{seed}.npz'
                with np.load(old,allow_pickle=False) as previous:
                    assert np.array_equal(data['ids'],previous['ids'])
                    reference={'exact':bool(np.array_equal(data['S8_repeats'],previous['D256_repeats'])),
                               'max_absolute_difference':float(np.max(np.abs(data['S8_repeats']-previous['D256_repeats'])))}
        run={'seed':seed,'regions':{},'S8_reference':reference}
        for region,values in errors.items():
            arms,point,samples,counts=_aggregate(rows,values,5000,2026)
            reports={key:_pair(arms,point,samples,a,b) for key,(a,b) in PAIRS.items()}
            run['regions'][region]={'mse':{a:float(point[i].mean()) for i,a in enumerate(arms)},'pairs':reports}
            for key,report in reports.items():
                p=report['primary'];lo,hi=p['ci95']['relative_improvement']
                records.append({'seed':seed,'region':region,'comparison':key,
                    'role':'primary' if region=='overall' and key=='S26_vs_H26' else 'secondary',
                    'mse_reference':p['mse_a'],'mse_treatment':p['mse_b'],
                    'gain_percent':100*p['relative_improvement'],
                    'ci95_low_percent':100*lo,'ci95_high_percent':100*hi})
        runs.append(run)
    summary={'split':split,'steps':6000,'seeds':list(SEEDS),'state_kib':KIB,'counts':counts,
        'primary':'S26_vs_H26 overall; both 26 KiB',
        'regions':{'overall':'all four B latent slices','first':'first B latent slice','later':'mean of remaining three B latent slices'},
        'statistics':'Fixed four-noise mean; window -> video -> source group -> class macro. Within-class source-group paired bootstrap: 5000 draws, seed2026. Across-three-seed summaries descriptive.',
        'mse':{reg:{arm:describe([r['regions'][reg]['mse'][arm] for r in runs]) for arm in ARMS} for reg in REGIONS},
        'gain_percent':{reg:{key:describe([100*r['regions'][reg]['pairs'][key]['primary']['relative_improvement'] for r in runs]) for key in PAIRS} for reg in REGIONS},
        'reference_checks':[{'seed':r['seed'],**r['S8_reference']} for r in runs if r['S8_reference'] is not None],
        'per_seed':records}
    (out/'summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    with (out/'summary.csv').open('w',newline='',encoding='utf-8') as f:
        writer=csv.DictWriter(f,fieldnames=list(records[0]));writer.writeheader();writer.writerows(records)
    plot(runs,summary,out)
    return summary


def plot(runs,summary,out):
    fig,axes=plt.subplots(1,2,figsize=(12,4.8),constrained_layout=True)
    values=[summary['mse']['overall'][a]['mean'] for a in ARMS]
    axes[0].bar(range(4),values,color=['#0072B2','#E69F00','#009E73','#CC79A7'])
    for i,v in enumerate(values):axes[0].text(i,v+.004,f'{v:.4f}',ha='center')
    axes[0].set(xticks=range(4),xticklabels=['State only\n8 KiB','Boundary only\n18 KiB','State + boundary\n26 KiB','State only\n26 KiB'],ylabel='Velocity MSE',title='Overall error (three-seed mean)',ylim=(0,max(values)*1.12))
    for j,r in enumerate(runs):
        for i,reg in enumerate(REGIONS):
            p=r['regions'][reg]['pairs']['S26_vs_H26']['primary'];lo,hi=100*np.asarray(p['ci95']['relative_improvement'])
            x=i+(j-1)*.13
            axes[1].vlines(x,lo,hi,color=f'C{j}')
            axes[1].plot(x,100*p['relative_improvement'],'o',color=f'C{j}',label=f"seed {r['seed']}" if i==0 else None)
    axes[1].axhline(0,color='grey',lw=.8)
    axes[1].set(xticks=range(3),xticklabels=['All B slices','First B slice','Later B slices'],ylabel='MSE reduction (%)',title='Hybrid over pure state: both 26 KiB')
    axes[1].legend(title='95% source-group CI',loc='best')
    fig.suptitle('Boundary latent and compressed history — final step 6000')
    fig.savefig(out/'comparison.png',dpi=180);plt.close(fig)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-root',type=Path)
    parser.add_argument('--results-dir',type=Path)
    parser.add_argument('--split',choices=('val','test'),default='test')
    args=parser.parse_args()
    out=args.results_dir or (args.run_root/f'boundary_state_results_{args.split}' if args.run_root else HERE/'results')
    if args.run_root:import_runs(args.run_root,out,args.split)
    result=analyze(out,args.split)
    print(json.dumps({'mse':result['mse'],'gain_percent':result['gain_percent']},indent=2))


if __name__=='__main__':main()
