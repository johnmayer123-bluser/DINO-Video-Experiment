"""Single-model, batch-one closed-loop latency and CUDA memory measurements.

No VAE is resident during generator measurements. Decode is measured separately
with the generator still resident. First-call and subsequent timings are retained.
"""
import argparse
import gc
import importlib.util
import json
from pathlib import Path
import time

import numpy as np
import torch


def module(name):
    spec = importlib.util.spec_from_file_location('delay_resource_' + name,
                                                Path(__file__).with_name(name + '.py'))
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


model_api, data = module('model'), module('data')


def synchronized_call(call):
    torch.cuda.synchronize()
    start = time.perf_counter()
    output = call()
    torch.cuda.synchronize()
    return output, (time.perf_counter() - start) * 1000


def storage_bytes(*tensors):
    unique = {t.untyped_storage().data_ptr(): t.untyped_storage().nbytes()
              for t in tensors}
    return sum(unique.values())


@torch.inference_mode()
def measure(args):
    model_api.width.setup()
    root = args.root
    output = root / 'runs' / f'delayed_memory_seed{args.seed}' / 'resources'
    output.mkdir(parents=True, exist_ok=True)
    candidates = data.load_dataset(root, 'val')
    rows = []
    for mode in ('keep', 'update'):
        rows.extend([r for r in candidates if r['mode'] == mode and r['variant'] == 0]
                    [:args.cases_per_mode])
    records, summaries = [], {}
    for arm in model_api.ARMS:
        gc.collect()
        torch.cuda.empty_cache()
        model = model_api.load_model(root, args.seed, arm, 'cuda')
        torch.cuda.synchronize()
        resident = torch.cuda.memory_allocated()
        last_output = None
        for delay in args.delays:
            for case_index, row in enumerate(rows):
                cached = data.episode_latents(root, row, delay)
                state = None
                initial_ms = 0.
                torch.cuda.reset_peak_memory_stats()
                for cue in cached['cues']:
                    history = cue[None].cuda().float()
                    def observe():
                        with model_api.precision(torch.device('cuda')):
                            next_state = model.observe(history, state)
                        return next_state, history[:, :, -1:].half().clone()
                    (state, boundary), cost = synchronized_call(observe)
                    initial_ms += cost
                    del history
                cue_peak = torch.cuda.max_memory_allocated()
                for position in range(delay + 1):
                    phase = int(position == delay)
                    persistent = storage_bytes(state, boundary)
                    torch.cuda.reset_peak_memory_stats()
                    def sample():
                        noise_seed = 880042 + case_index * 100003 + delay * 101 + position
                        x = torch.randn((1, 16, 4, 24, 24),
                                        generator=torch.Generator().manual_seed(noise_seed)).cuda()
                        for step in range(args.steps):
                            t = torch.full((1,), step / args.steps, device='cuda')
                            with model_api.precision(torch.device('cuda')):
                                velocity = model.predict(boundary, x, t, state, phase)
                            x = x + velocity.float() / args.steps
                        return x
                    generated, sample_ms = synchronized_call(sample)
                    update_ms = 0.
                    if not phase:
                        def update():
                            with model_api.precision(torch.device('cuda')):
                                next_state = model.observe(generated, state)
                            return next_state, generated[:, :, -1:].half().clone()
                        (state, boundary), update_ms = synchronized_call(update)
                    records.append({'arm': arm, 'id': row['id'], 'mode': row['mode'],
                                    'delay': delay, 'position': position, 'phase': phase,
                                    'sample_ms': sample_ms, 'update_ms': update_ms,
                                    'sample_plus_update_ms': sample_ms + update_ms,
                                    'initial_cues_ms': initial_ms if position == 0 else None,
                                    'initial_cues_peak_bytes': cue_peak if position == 0 else None,
                                    'persistent_cache_bytes': persistent,
                                    'peak_allocated_bytes': torch.cuda.max_memory_allocated(),
                                    'peak_reserved_bytes': torch.cuda.max_memory_reserved()})
                    last_output = generated.cpu()
                    del generated
                del state, boundary, cached
        subset = [r for r in records if r['arm'] == arm]
        steady = subset[1:]
        summaries[arm] = {'parameters': sum(p.numel() for p in model.parameters()),
                          'parameter_bytes': sum(p.numel() * p.element_size() for p in model.parameters()),
                          'model_resident_allocated_bytes': resident,
                          'first_sample_ms': subset[0]['sample_ms'],
                          'later_sample_median_ms': float(np.median([r['sample_ms'] for r in steady])),
                          'hidden_update_median_ms': float(np.median([r['update_ms'] for r in steady if not r['phase']])),
                          'peak_generator_allocated_bytes': max(
                              max(r['peak_allocated_bytes'], r['initial_cues_peak_bytes'] or 0)
                              for r in subset),
                          'cache_bytes': sorted({r['persistent_cache_bytes'] for r in subset})}
        if args.decode:
            from diffusers import AutoencoderKLWan
            vae = AutoencoderKLWan.from_pretrained(root / 'models/wan-vae',
                      torch_dtype=torch.float32, local_files_only=True).cuda().eval()
            mean = torch.tensor(vae.config.latents_mean, device='cuda').view(1, 16, 1, 1, 1)
            std = torch.tensor(vae.config.latents_std, device='cuda').view(1, 16, 1, 1, 1)
            z = last_output.cuda()
            decode_times = []
            torch.cuda.reset_peak_memory_stats()
            for _ in range(6):
                def decode():
                    return vae.decode(torch.cat((z[:, :, :1], z), 2) * std + mean).sample
                decoded, duration = synchronized_call(decode)
                decode_times.append(duration)
                del decoded
            summaries[arm]['vae_decode'] = {
                'first_ms': decode_times[0], 'later_median_ms': float(np.median(decode_times[1:])),
                'all_ms': decode_times,
                'combined_model_vae_peak_allocated_bytes': torch.cuda.max_memory_allocated(),
                'method': '17 frames from duplicated generated first latent; retain last 16 frames'}
            del vae, mean, std, z
        del model, last_output
        gc.collect()
        torch.cuda.empty_cache()
        print(json.dumps({'arm': arm, **summaries[arm]}), flush=True)
        (output / 'summary.json').write_text(json.dumps(summaries, indent=2), encoding='utf-8')
    metadata = {'device': torch.cuda.get_device_name(), 'torch': torch.__version__,
                'gpu_total_memory_bytes': torch.cuda.get_device_properties(0).total_memory,
                'seed': args.seed, 'batch_size': 1, 'sampler': 'Euler', 'steps': args.steps,
                'delays': args.delays, 'cases': [r['id'] for r in rows],
                'timing': 'CUDA-synchronized wall time; sampling includes CPU Gaussian creation and transfer',
                'memory': 'one FP32-weight model; BF16 autocast; no VAE or other arms resident for generator measurement',
                'cache': 'unique underlying storage bytes of FP16 representation and boundary, no full chunk retained',
                'scope': 'generator timings include encoder/update separately; VAE decode separate; data preprocessing excluded',
                'history_scaling': 'all three arms have fixed history storage; this is not a comparison to unbounded-history attention'}
    (output / 'metadata.json').write_text(json.dumps(metadata, indent=2), encoding='utf-8')
    (output / 'cases.json').write_text(json.dumps(records, indent=2), encoding='utf-8')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=model_api.ROOT)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--steps', type=int, default=32)
    parser.add_argument('--delays', nargs='+', type=int, default=[1, 3, 6])
    parser.add_argument('--cases-per-mode', type=int, default=2)
    parser.add_argument('--decode', action='store_true')
    measure(parser.parse_args())
