"""Independent batch-one causal-rollout resource measurements; no VAE resident.

python resources.py --device cuda --repeats 3
Every timed run includes cue encoding, generated blank updates and one return.
"""
import argparse
import gc
from pathlib import Path

import numpy as np
import torch

from common import WORK_ROOT, write_json
from evaluate import balanced_subset, chunk_noise, sample_chunk, timed, unique_storage_kib


@torch.inference_mode()
def run_rollout(model, cue, delay, pair_id, device):
    from model import precision
    with precision(torch.device(device)):
        state = model.observe(cue)
    boundary = cue[:, :, -1:].half().clone()
    for j in range(1,delay+1):
        blank = sample_chunk(model,boundary,state,chunk_noise(pair_id,j),phase=0)
        with precision(torch.device(device)):
            state = model.observe(blank,state)
        boundary = blank[:, :, -1:].half().clone()
    returned = sample_chunk(model,boundary,state,chunk_noise(pair_id),phase=1)
    return state,boundary,returned


@torch.inference_mode()
def main():
    from data import load_rows, load_pair
    from model import ARMS, load_model
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--device',default='cuda')
    parser.add_argument('--seed',type=int,default=42,choices=[42])
    parser.add_argument('--delays',type=int,nargs='+',default=[1,4,8,16,32])
    parser.add_argument('--repeats',type=int,default=3)
    parser.add_argument('--output',type=Path)
    args = parser.parse_args()
    if args.repeats<1 or min(args.delays)<1 or max(args.delays)>32:
        parser.error('repeats positive; delays 1..32')
    torch.set_num_threads(4)
    output = args.output or WORK_ROOT/'results/seed42/resources_batch1.json'
    row = balanced_subset(load_rows('test'),1)[0]
    pair_id = str(row['pair_id'])
    results = {'seed':42,'batch_size':1,'euler_steps':32,'repeats':args.repeats,
        'scope':'cue observation + delay generated blank chunks and their observations + one generated return; no VAE',
        'informative_persistent_condition_kib':26,
        'persistent_scope':'FP16 [1,16,256] state + owned [1,16,1,24,24] boundary; excludes cue, sampling/work buffers and model parameters',
        'noise':'identical per pair/chunk across arms; return shared across delays',
        'arms':{}}
    for arm in ARMS:
        gc.collect()
        if torch.device(args.device).type=='cuda':
            torch.cuda.empty_cache()
        model = load_model(arm,args.device,step=6000,seed=42)
        cue = load_pair(pair_id)['cue'][0:1].to(args.device).float()
        # Complete warmup, including blank and return samplers, before any timing.
        warm = run_rollout(model,cue,max(args.delays),pair_id,args.device)
        del warm
        for _ in range(3):
            warm = run_rollout(model,cue,1,pair_id,args.device)
            del warm
        arm_results = {}
        for delay in sorted(set(args.delays)):
            durations = []
            if torch.device(args.device).type=='cuda':
                torch.cuda.reset_peak_memory_stats(args.device)
            for _ in range(args.repeats):
                result,ms = timed(lambda:run_rollout(model,cue,delay,pair_id,args.device),args.device)
                durations.append(ms)
                actual = unique_storage_kib(result[:2])
                del result
            median = float(np.median(durations))
            arm_results[str(delay)] = {'delay':delay,'total_samples_ms':durations,'total_median_ms':median,
                'generated_chunks':delay+1,'amortized_ms_per_generated_chunk':median/(delay+1),
                'amortized_scope':'includes cue encoding and all updates; not isolated single sampler latency',
                'actual_persistent_tensor_kib':actual,
                'peak_allocated_bytes':torch.cuda.max_memory_allocated(args.device) if torch.device(args.device).type=='cuda' else None,
                'peak_reserved_bytes':torch.cuda.max_memory_reserved(args.device) if torch.device(args.device).type=='cuda' else None}
            print(f'batch1 {arm} delay={delay}: total={median:.1f}ms; per chunk={median/(delay+1):.1f}ms',flush=True)
        results['arms'][arm] = {'warmup_complete_rollouts':4,
            'warmup_delays':[max(args.delays),1,1,1],'vae_resident':False,'delays':arm_results}
        del model,cue
    write_json(output,results)
    print(f'Saved {output}',flush=True)


if __name__=='__main__':
    main()
