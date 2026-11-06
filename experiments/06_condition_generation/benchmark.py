"""Measure warmed inference timing in a balanced N/C/C/N model order."""
import argparse
import gc
import json
import statistics
import torch
from common import WORK_ROOT, write_json
from data import load_pair
from model import load_model, precision
from evaluate import pair_noise, sample_chunk, timed


@torch.inference_mode()
def main():
    argparse.ArgumentParser(description=__doc__).parse_args()
    torch.set_num_threads(4)
    root = WORK_ROOT / 'results/seed42'
    row = json.loads((root / 'manifest.json').read_text())['rows'][0]
    item = load_pair(row['pair_id'])
    measurements = []
    for arm in ('N', 'C', 'C', 'N'):
        gc.collect()
        torch.cuda.empty_cache()
        model = load_model(arm)
        cue = item['cue'][:1].cuda().float()
        boundary = item['boundary'][None].cuda().half()
        noise = pair_noise(row['pair_id'])
        def encode():
            with precision('cuda'):
                return model.observe(cue)
        state = encode() if arm == 'C' else torch.zeros(1,16,256,device='cuda',dtype=torch.float16)
        for _ in range(5):
            output = sample_chunk(model, boundary, state, noise)
            del output
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        encoding, sampling = [], []
        for _ in range(10):
            if arm == 'C':
                state, ms = timed(encode, 'cuda')
                encoding.append(ms)
            output, ms = timed(lambda: sample_chunk(model, boundary, state, noise), 'cuda')
            sampling.append(ms)
            del output
        record = {'arm': arm, 'cue_encoding_samples_ms': encoding, 'euler32_samples_ms': sampling,
                  'peak_allocated_bytes': torch.cuda.max_memory_allocated(),
                  'peak_reserved_bytes': torch.cuda.max_memory_reserved()}
        measurements.append(record)
        print(arm, 'median sampling ms', statistics.median(sampling), flush=True)
        del model, cue, boundary, state
    write_json(root / 'timing_recheck.json', {'order': ['N','C','C','N'], 'warmup_per_load': 5,
               'timed_per_load': 10, 'batch_size': 1, 'measurements': measurements})
    resources = json.loads((root / 'resources.json').read_text())
    for arm in ('N','C'):
        result = resources['arms'][arm]
        result['initial_timing'] = {k: result[k] for k in
            ('cue_encoding_ms','euler32_ms','cue_encoding_samples_ms','euler32_samples_ms')}
        records = [r for r in measurements if r['arm'] == arm]
        enc = [v for r in records for v in r['cue_encoding_samples_ms']]
        samples = [v for r in records for v in r['euler32_samples_ms']]
        result.update(cue_encoding_ms=statistics.median(enc) if enc else 0.,
                      euler32_ms=statistics.median(samples),
                      cue_encoding_samples_ms=enc, euler32_samples_ms=samples,
                      timing_repeats=len(samples), warmup_rollouts=5,
                      timing_protocol='two loads per arm, N/C/C/N order, 5 warmups and 10 timed rollouts per load',
                      peak_allocated_bytes=max(r['peak_allocated_bytes'] for r in records),
                      peak_reserved_bytes=max(r['peak_reserved_bytes'] for r in records))
    write_json(root / 'resources.json', resources)


if __name__ == '__main__':
    main()
