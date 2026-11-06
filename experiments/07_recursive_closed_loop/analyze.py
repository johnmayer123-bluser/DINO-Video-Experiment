"""CPU-only reporting for completed Q7 artifacts; never imports torch/models.

python analyze.py
Writes results/seed42/report.json and comparison.png without modifying inputs.
"""
import argparse
from collections import defaultdict
import json
import math
from pathlib import Path
import statistics

from common import WORK_ROOT, write_json


ARMS = ('F', 'L', 'R')
LABELS = {'F':'F Fixed cue', 'L':'L Latest chunk', 'R':'R Recursive'}
COLORS = {'F':'#4466aa', 'L':'#dd8833', 'R':'#339966'}


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def number(value, name):
    if isinstance(value, bool) or not isinstance(value, (int,float)) or not math.isfinite(value):
        raise ValueError(f'{name}: expected finite number, got {value!r}')
    return value


def analyze(results, runs):
    summary = read_json(results/'summary.json')
    resources = read_json(results/'resources_batch1.json')
    leakage = read_json(results/'blank_leakage.json')
    if summary['seed'] != 42 or resources['seed'] != 42 or resources['batch_size'] != 1:
        raise ValueError('Expected seed42 and independent batch_size=1 resources')
    metrics = summary['metrics']
    delays = sorted(int(key.split('/')[2][1:]) for key in metrics
                    if key.startswith('R/closed/') and key.endswith('/normal'))
    if not delays or len(delays) != len(set(delays)):
        raise ValueError('Missing/duplicate R closed-loop delays')
    report = {'seed':42,'pairs':summary['pairs'],'delays':delays,
        'joint_accuracy_definition':'mean per-variant last8-frame color AND shape correctness',
        'paired_success_definition':'both variants strictly >0.5 last8-frame correctness',
        'timing_definition':'median complete batch1 rollout divided by delay+1 generated chunks; includes cue and state updates',
        'arms':{},'boundary_interventions':{},'blank_leakage':{}}
    for arm in ARMS:
        training = read_json(runs/f'resource_{arm}.json')
        if training['arm'] != arm or training['step'] != 6000:
            raise ValueError(f'Unexpected training arm/step for {arm}')
        arm_report = {'label':LABELS[arm], 'training':{
            'hours':number(training['training_seconds'],f'{arm} training seconds')/3600,
            'peak_allocated_mib':number(training['peak_allocated_bytes'],f'{arm} training peak')/2**20},
            'delays':{}}
        for delay in delays:
            item = {}
            for history in ('truth','closed'):
                group = metrics[f'{arm}/{history}/d{delay}/normal']
                item[history] = {key:number(group[key],f'{arm}/{history}/{delay}/{key}')
                    for key in ('n','joint_accuracy','color_accuracy','shape_accuracy','spatial_iou',
                                'paired_both_correct','paired_control_following')}
            raw = resources['arms'][arm]['delays'][str(delay)]
            if raw['delay'] != delay or raw['generated_chunks'] != delay+1:
                raise ValueError(f'Resource delay/chunk mismatch: {arm} d{delay}')
            item['batch1'] = {
                'total_median_ms':number(raw['total_median_ms'],f'{arm}/{delay}/total'),
                'amortized_ms_per_generated_chunk':number(raw['amortized_ms_per_generated_chunk'],f'{arm}/{delay}/per chunk'),
                'peak_allocated_mib':number(raw['peak_allocated_bytes'],f'{arm}/{delay}/peak')/2**20,
                'actual_persistent_tensor_kib':number(raw['actual_persistent_tensor_kib'],f'{arm}/{delay}/condition')}
            arm_report['delays'][str(delay)] = item
        report['arms'][arm] = arm_report
    for delay in (1,32):
        if delay not in delays:
            continue
        groups = {condition:metrics[f'R/closed/d{delay}/{condition}']
                  for condition in ('normal','clamp_boundary','clamp_boundary_zero')}
        report['boundary_interventions'][str(delay)] = {
            condition:{key:number(group[key],f'R/{delay}/{condition}/{key}')
                       for key in ('joint_accuracy','color_accuracy','shape_accuracy','paired_both_correct')}
            for condition,group in groups.items()}
        report['boundary_interventions'][str(delay)]['joint_differences'] = {
            'normal_minus_clamp_boundary':groups['normal']['joint_accuracy']-groups['clamp_boundary']['joint_accuracy'],
            'clamp_boundary_minus_clamp_boundary_zero':groups['clamp_boundary']['joint_accuracy']-groups['clamp_boundary_zero']['joint_accuracy']}
    leak_groups = defaultdict(list)
    for row in leakage:
        if row['arm'] not in ARMS or row['delay'] not in delays:
            raise ValueError(f'Unexpected leakage arm/delay: {row}')
        leak_groups[(row['arm'],row['delay'])].append(row)
    for arm in ARMS:
        report['blank_leakage'][arm] = {}
        for delay in delays:
            cases = leak_groups[(arm,delay)]
            if not cases:
                raise ValueError(f'Missing blank leakage observations: {arm} d{delay}')
            report['blank_leakage'][arm][str(delay)] = {'n_variants':len(cases),
                **{key:statistics.mean(number(row[key],f'{arm}/{delay}/{key}') for row in cases)
                   for key in ('saturated_area_fraction','last8_saturated_area_fraction','visible_presence_fraction')}}
    return report


def plot(report, path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    delays = report['delays']
    fig,axes = plt.subplots(2,2,figsize=(11,8),constrained_layout=True)
    for arm in ARMS:
        data = report['arms'][arm]['delays']
        for ax,history in zip(axes[0],('truth','closed')):
            ax.plot(delays,[data[str(d)][history]['joint_accuracy'] for d in delays],
                    marker='o',label=LABELS[arm],color=COLORS[arm])
        axes[1,0].plot(delays,[data[str(d)]['batch1']['amortized_ms_per_generated_chunk'] for d in delays],
                       marker='o',label=LABELS[arm],color=COLORS[arm])
        axes[1,1].plot(delays,[data[str(d)]['batch1']['peak_allocated_mib'] for d in delays],
                       marker='o',label=LABELS[arm],color=COLORS[arm])
    axes[0,0].set(title='True blank history: return quality',ylabel='Joint accuracy (last 8 frames)',ylim=(-.03,1.03))
    axes[0,1].set(title='Generated closed loop: return quality',ylabel='Joint accuracy (last 8 frames)',ylim=(-.03,1.03))
    axes[1,0].set(title='Batch 1 rollout latency / generated chunk',ylabel='Amortized milliseconds / chunk')
    axes[1,1].set(title='Batch 1 generator peak allocated memory',ylabel='Peak allocated MiB (VAE absent)')
    for ax in axes.flat:
        ax.set_xlabel('Number of blank chunks before return')
        ax.set_xticks(delays)
        ax.grid(alpha=.25)
        ax.legend(fontsize=9)
    fig.suptitle(f'Q7 single seed 42 | {report["pairs"]} paired scenes',fontsize=14)
    fig.savefig(path,dpi=180)
    plt.close(fig)


def print_tables(report):
    print('\n质量与推理资源（联合正确率为最后8帧均值；时间为 batch1 全流程摊销）')
    print('模型\t延迟块\t真历史联合%\t闭环联合%\t闭环成对成功%\t毫秒/块\t峰值MiB\t条件KiB')
    for arm in ARMS:
        for delay in report['delays']:
            item = report['arms'][arm]['delays'][str(delay)]
            truth,closed,res = item['truth'],item['closed'],item['batch1']
            print(f'{arm}\t{delay}\t{truth["joint_accuracy"]*100:.2f}\t{closed["joint_accuracy"]*100:.2f}\t'
                  f'{closed["paired_both_correct"]*100:.2f}\t{res["amortized_ms_per_generated_chunk"]:.2f}\t'
                  f'{res["peak_allocated_mib"]:.1f}\t{res["actual_persistent_tensor_kib"]:.2f}')
    print('\n训练资源\n模型\t训练小时\t训练峰值MiB')
    for arm in ARMS:
        item = report['arms'][arm]['training']
        print(f'{arm}\t{item["hours"]:.3f}\t{item["peak_allocated_mib"]:.1f}')
    print('\nR 的边界干预（联合正确率）\n延迟块\t正常%\t固定空白边界%\t固定边界且清零状态%')
    for delay,item in report['boundary_interventions'].items():
        print(f'{delay}\t{item["normal"]["joint_accuracy"]*100:.2f}\t'
              f'{item["clamp_boundary"]["joint_accuracy"]*100:.2f}\t{item["clamp_boundary_zero"]["joint_accuracy"]*100:.2f}')
    print('\n闭环最后空白块泄漏诊断（所选前2对；可见判据为饱和面积阈值）\n模型\t延迟块\t最后8帧饱和面积%\t可见帧%')
    for arm in ARMS:
        for delay,item in report['blank_leakage'][arm].items():
            print(f'{arm}\t{delay}\t{item["last8_saturated_area_fraction"]*100:.4f}\t{item["visible_presence_fraction"]*100:.2f}')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results',type=Path,default=WORK_ROOT/'results/seed42')
    parser.add_argument('--runs',type=Path,
                        help='Training resources directory; defaults to an active local run, then RESULTS/training')
    args = parser.parse_args()
    runs = args.runs
    if runs is None:
        local_run = WORK_ROOT/'runs/seed42'
        runs = local_run if local_run.exists() else args.results/'training'
    report = analyze(args.results,runs)
    plot(report,args.results/'comparison.png')
    write_json(args.results/'report.json',report)
    print_tables(report)
    print(f'\n已保存：{args.results / "report.json"}\n已保存：{args.results / "comparison.png"}')


if __name__=='__main__':
    main()
