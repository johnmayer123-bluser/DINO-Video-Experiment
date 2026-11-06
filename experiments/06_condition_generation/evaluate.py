"""Small paired keep-condition evaluation; model and VAE never coexist on GPU.

python evaluate.py --pairs 60 --seed 42 --device cuda --preview-pairs 4
Use --pairs 2 --output results/smoke for a short end-to-end smoke run.
"""
import argparse
from collections import defaultdict
import csv
import gc
import hashlib
from pathlib import Path
import time

import numpy as np
from PIL import Image, ImageDraw
import torch

from common import REPO_ROOT, WORK_ROOT, write_json


def rgb_float(frames):
    x = np.asarray(frames, dtype=np.float32)
    return x / 255. if x.max(initial=0) > 1.5 else x


def visible_mask(frames):
    x = rgb_float(frames)
    return (x.max(-1) - x.min(-1) > .22) & (x.max(-1) > .35)


def bbox_shape(mask, size=40):
    """Translation/scale invariant silhouette; no target-position constraint."""
    y, x = np.where(mask)
    if len(x) < 12:
        return None
    crop = mask[y.min():y.max()+1, x.min():x.max()+1]
    return np.asarray(Image.fromarray(crop).resize((size, size), Image.Resampling.NEAREST), bool)


def iou(a, b):
    return float(np.logical_and(a, b).sum() / max(1, np.logical_or(a, b).sum()))


def largest_component(mask):
    """Four-connected row-run labeling; avoids a new scipy/cv2 dependency."""
    runs, parents, areas = [], [], []
    previous = []
    def root(i):
        while parents[i] != i:
            parents[i] = parents[parents[i]]
            i = parents[i]
        return i
    for y, line in enumerate(mask):
        edges = np.diff(np.pad(line.astype(np.int8), (1,1)))
        current = []
        for start, stop in zip(np.flatnonzero(edges==1),np.flatnonzero(edges==-1)):
            index = len(parents)
            parents.append(index)
            areas.append(int(stop-start))
            runs.append((y,int(start),int(stop)))
            current.append(index)
            for old in previous:
                _, a, b = runs[old]
                if a < stop and start < b:
                    p, q = root(index), root(old)
                    if p != q:
                        parents[q] = p
                        areas[p] += areas[q]
        previous = current
    clean = np.zeros_like(mask)
    if runs:
        winner = max((i for i in range(len(runs)) if root(i)==i),key=lambda i:areas[i])
        for i,(y,start,stop) in enumerate(runs):
            if root(i)==winner:
                clean[y,start:stop] = True
    return clean


class Scorer:
    """Twelve tiny renderer references, reused for every last-eight-frame score.

    Presence and color have no positional gate. Shape uses bounding-box aligned
    silhouettes; global spatial IoU is a separate diagnostic. Classification
    coverage and accuracy including missing/rejected frames are both reported.
    """
    def __init__(self, reference_row):
        from data import render_chunk
        self.palette, self.shapes = {}, {}
        self.scene_shapes = {}
        for color in range(4):
            for shape in range(3):
                row = dict(reference_row, color_id=color, shape_id=shape, attr=color*3+shape)
                frames, masks = render_chunk(row, kind='return', return_mask=True)
                pixels = rgb_float(frames)[np.asarray(masks, bool)]
                if not len(pixels):
                    raise ValueError('Renderer reference has no visible object')
                self.palette[color] = np.median(pixels, axis=0)
                self.shapes[shape] = bbox_shape(np.asarray(masks[-1], bool))

    def templates(self, row):
        from data import render_chunk
        key = tuple(sorted(row['scene'].items()))
        if key not in self.scene_shapes:
            self.scene_shapes[key] = {
                shape:[bbox_shape(mask.astype(bool)) for mask in render_chunk(
                    dict(row,shape_id=shape,attr=int(row['color_id'])*3+shape),
                    kind='return',return_mask=True)[1][-8:]] for shape in range(3)}
        return self.scene_shapes[key]

    def score(self, frames, row, target_mask):
        frames = rgb_float(frames)[-8:]
        masks = visible_mask(frames)
        target_mask = np.asarray(target_mask, bool)[-8:]
        templates = self.templates(row)
        frame_cases = []
        for frame_index,(frame, mask, truth) in enumerate(zip(frames, masks, target_mask)):
            raw_mask = mask
            mask = largest_component(mask)
            # A fixed tiny minimum area rejects isolated colored codec noise.
            present = max(12, round(mask.size*.0004)) <= int(mask.sum()) <= mask.size*.30
            color = shape = None
            distance = shape_iou = None
            if present:
                observed = np.median(frame[mask], axis=0)
                distances = {c: float(np.linalg.norm(observed-p)) for c, p in self.palette.items()}
                candidate = min(distances, key=distances.get)
                distance = distances[candidate]
                if distance <= .65:
                    color = candidate
                silhouette = bbox_shape(mask)
                if silhouette is not None:
                    scores = {s:iou(silhouette, sequence[frame_index])
                              for s,sequence in templates.items() if sequence[frame_index] is not None}
                    candidate = max(scores, key=scores.get)
                    shape_iou = scores[candidate]
                    if shape_iou >= .40:
                        shape = candidate
            frame_cases.append({'presence': float(present),
                                'color_classifiable': float(color is not None),
                                'shape_classifiable': float(shape is not None),
                                'color_accuracy': float(color == int(row['color_id'])),
                                'shape_accuracy': float(shape == int(row['shape_id'])),
                                'joint_accuracy': float(color == int(row['color_id']) and shape == int(row['shape_id'])),
                                'spatial_iou': iou(raw_mask, truth),
                                'predicted_color': color, 'predicted_shape': shape,
                                'color_distance': distance, 'aligned_shape_iou': shape_iou})
        metrics = {key: float(np.mean([f[key] for f in frame_cases])) for key in METRICS}
        for kind in ('color', 'shape'):
            accepted = [f for f in frame_cases if f[f'{kind}_classifiable']]
            metrics[f'{kind}_accuracy_when_classifiable'] = (
                float(np.mean([f[f'{kind}_accuracy'] for f in accepted])) if accepted else None)
        return {**metrics, 'frames': frame_cases}


METRICS = ('presence', 'color_classifiable', 'shape_classifiable', 'color_accuracy',
           'shape_accuracy', 'joint_accuracy', 'spatial_iou')


def balanced_subset(rows, pairs):
    groups = defaultdict(list)
    for row in rows:
        groups[str(row['pair_id'])].append(row)
    buckets = defaultdict(list)
    for pair_id, members in sorted(groups.items()):
        if len(members) != 2 or {int(r['variant']) for r in members} != {0, 1}:
            raise ValueError(f'Incomplete counterfactual pair {pair_id}')
        signature = (str(members[0]['pair_change']),
                     tuple(sorted((int(r['color_id']), int(r['shape_id'])) for r in members)))
        buckets[signature].append(pair_id)
    chosen = []
    signatures = {change:sorted(k for k in buckets if k[0]==change)
                  for change in sorted({k[0] for k in buckets})}
    cursors = defaultdict(int)
    limit = min(pairs, len(groups))
    while len(chosen) < limit:
        for change, keys in signatures.items():
            available = [k for k in keys if buckets[k]]
            if available and len(chosen) < limit:
                key = available[cursors[change] % len(available)]
                chosen.append(buckets[key].pop(0))
                cursors[change] += 1
    return [r for p in chosen for r in sorted(groups[p], key=lambda r: int(r['variant']))]


def pair_noise(pair_id):
    key = hashlib.sha256(f'2026|{pair_id}'.encode()).digest()
    generator = torch.Generator(device='cpu').manual_seed(int.from_bytes(key[:8], 'little') % (2**63-1))
    return torch.randn((1, 16, 4, 24, 24), generator=generator)


@torch.inference_mode()
def sample_chunk(model, boundary, state, noise):
    from model import precision
    x = noise.to(boundary.device).float().clone()
    for step in range(32):
        t = torch.full((len(x),), step/32, device=x.device)
        with precision(x.device):
            velocity = model.predict(boundary, x, t, state, phase=1)
        x.add_(velocity.float(), alpha=1/32)
    return x


class IndependentDecoder:
    def __init__(self, device):
        from diffusers import AutoencoderKLWan
        self.device = torch.device(device)
        self.vae = AutoencoderKLWan.from_pretrained(
            Path(REPO_ROOT)/'models/wan-vae', torch_dtype=torch.float32,
            local_files_only=True).eval().requires_grad_(False).to(device)
        self.mean = torch.tensor(self.vae.config.latents_mean, device=device).view(1,16,1,1,1)
        self.std = torch.tensor(self.vae.config.latents_std, device=device).view(1,16,1,1,1)

    @torch.inference_mode()
    def __call__(self, chunks):
        x = chunks.to(self.device).float()
        x = torch.cat((x[:, :, :1], x), dim=2)
        decoded = self.vae.decode(x*self.std+self.mean).sample
        return decoded[:, :, -16:].add(1).div(2).clamp(0,1).permute(0,2,3,4,1).cpu().numpy()


def synchronize(device):
    if torch.device(device).type == 'cuda':
        torch.cuda.synchronize(device)


def timed(function, device):
    synchronize(device)
    start = time.perf_counter()
    value = function()
    synchronize(device)
    return value, (time.perf_counter()-start)*1000


def unique_storage_kib(tensors):
    """Count actual underlying storage once, including any view backing space."""
    storages = {}
    for tensor in tensors:
        storage = tensor.untyped_storage()
        key = (str(tensor.device),storage.data_ptr())
        storages[key] = storage.nbytes()
    return sum(storages.values())/1024


@torch.inference_mode()
def generate_latents(rows, args):
    from data import load_pair
    from model import ARMS, load_model, precision
    cached, resources = [], {}
    pair_groups = [rows[i:i+2] for i in range(0, len(rows), 2)]
    for arm in ARMS:
        gc.collect()
        if torch.device(args.device).type == 'cuda':
            torch.cuda.empty_cache()
        model = load_model(arm, device=args.device, step=6000, seed=args.seed)
        # Measure batch=1, after exactly one complete warmup, without a VAE.
        first = load_pair(rows[0]['pair_id'])
        cue = first['cue'][0:1].to(args.device).float() if arm == 'C' else None
        boundary = first['boundary'].unsqueeze(0).to(args.device).half()
        def encode(observation=cue):
            if arm == 'N':
                return torch.zeros((1,16,256),device=args.device,dtype=torch.float16)
            with precision(torch.device(args.device)):
                return model.observe(observation)
        state = encode()
        warm = sample_chunk(model, boundary, state, pair_noise(rows[0]['pair_id']))
        del warm, state
        synchronize(args.device)
        if torch.device(args.device).type == 'cuda':
            torch.cuda.reset_peak_memory_stats(args.device)
        encoding_samples = []
        for repeat in range(3):
            if repeat:
                del measured_state
            measured_state, state_ms = timed(encode, args.device)
            encoding_samples.append(state_ms)
        state_ms = float(np.median(encoding_samples))
        durations = []
        for _ in range(3):
            measured, duration = timed(lambda: sample_chunk(model, boundary, measured_state,
                                      pair_noise(rows[0]['pair_id'])), args.device)
            durations.append(duration)
            del measured
        sample_ms = float(np.median(durations))
        if arm == 'N':
            state_ms = 0.
        resource = {'batch_size': 1, 'warmup_rollouts': 1, 'cue_encoding_ms': state_ms,
                    'timing_repeats':3, 'euler32_samples_ms':durations,
                    'cue_encoding_samples_ms':encoding_samples if arm=='C' else [],
                    'cue_encoding_skipped':arm=='N',
                    'euler32_ms': sample_ms,
                    'informative_condition_kib': 18 if arm == 'N' else 26,
                    'actual_persistent_tensor_kib': unique_storage_kib((boundary,measured_state)),
                    'zero_placeholder_kib': unique_storage_kib((measured_state,)) if arm=='N' else 0,
                    'persistent_tensor_scope': 'batch1 FP16 boundary and state, unique backing storage; excludes cue input, rollout buffers and model parameters',
                    'vae_resident': False, 'precision': 'bfloat16 autocast on CUDA; fp32 integration',
                    'peak_allocated_bytes': None, 'peak_reserved_bytes': None}
        if torch.device(args.device).type == 'cuda':
            resource.update(peak_allocated_bytes=torch.cuda.max_memory_allocated(args.device),
                            peak_reserved_bytes=torch.cuda.max_memory_reserved(args.device))
        resources[arm] = resource
        del cue, boundary, measured_state, first, encode
        for pair_index,members in enumerate(pair_groups):
            item = load_pair(members[0]['pair_id'])
            states = []
            for variant in (0, 1):
                observation = item['cue'][variant:variant+1].to(args.device).float() if arm == 'C' else None
                if arm == 'N':
                    encoded = torch.zeros((1,16,256),device=args.device,dtype=torch.float16)
                else:
                    with precision(torch.device(args.device)):
                        encoded = model.observe(observation)
                states.append(encoded)
            boundary = item['boundary'][None].to(args.device).half()
            noise = pair_noise(members[0]['pair_id'])
            generated = {}
            if arm == 'N':
                generated['shared'] = sample_chunk(model,boundary,states[0],noise).half().cpu()
            else:
                for variant in (0,1):
                    generated[f'normal{variant}'] = sample_chunk(model,boundary,states[variant],noise).half().cpu()
                generated['zero'] = sample_chunk(model,boundary,torch.zeros_like(states[0]),noise).half().cpu()
            for row in members:
                variant = int(row['variant'])
                if arm == 'N':
                    cached.append((row,arm,'normal',generated['shared']))
                else:
                    cached.extend([(row,arm,'normal',generated[f'normal{variant}']),
                                   (row,arm,'zero',generated['zero']),
                                   (row,arm,'swap',generated[f'normal{1-variant}'])])
            del item, states, boundary, observation, encoded, generated
            if (pair_index+1)%10==0 or pair_index+1==len(pair_groups):
                print(f'generated {arm} pairs {pair_index+1}/{len(pair_groups)}', flush=True)
        del model
        gc.collect()
        if torch.device(args.device).type == 'cuda':
            torch.cuda.empty_cache()
    unique = {id(x):x for _,_,_,x in cached}
    size = sum(x.numel()*x.element_size() for x in unique.values())
    return cached, {'arms': resources, 'cpu_generated_latent_bytes': size}


def aggregate(cases):
    result = {}
    groups = defaultdict(list)
    for case in cases:
        groups[f'{case["arm"]}/{case["condition"]}'].append(case)
    for key, members in groups.items():
        result[key] = {'n': len(members), **{k: float(np.mean([m[k] for m in members])) for k in METRICS}}
        for kind in ('color','shape'):
            coverage = sum(m[f'{kind}_classifiable'] for m in members)
            result[key][f'{kind}_accuracy_when_classifiable'] = (
                sum(m[f'{kind}_accuracy'] for m in members)/coverage if coverage else None)
        pairs = defaultdict(list)
        for case in members:
            pairs[case['pair_id']].append(case)
        complete = [p for p in pairs.values() if len(p) == 2]
        result[key]['paired_both_correct'] = float(np.mean([
            all(m['joint_accuracy'] > .5 for m in p) for p in complete])) if complete else None
        result[key]['paired_control_following'] = float(np.mean([
            all(m[f'{m["pair_change"]}_accuracy'] > .5 for m in p)
            for p in complete])) if complete else None
    return result


def preview_pair_ids(rows, count):
    chosen = []
    for change in ('color', 'shape'):
        candidates = sorted({str(r['pair_id']) for r in rows if r['pair_change'] == change})
        chosen.extend(candidates[:2])
    for row in rows:
        if len(chosen) >= count:
            break
        if str(row['pair_id']) not in chosen:
            chosen.append(str(row['pair_id']))
    return set(chosen[:count])


def export_preview(output, row, generated):
    import av
    from data import render_chunk
    cue, target = render_chunk(row, kind='cue'), render_chunk(row, kind='return')
    panels = [cue, target, generated['N'], generated['C']]
    side = int(target.shape[1])
    height = side+40
    labels = ['cue TRUE OBSERVATION', 'return TRUE TARGET', 'N GENERATED', 'C GENERATED']
    def compose(t):
        canvas = Image.new('RGB', (4*side, height), 'white')
        draw = ImageDraw.Draw(canvas)
        for j, (frames, label) in enumerate(zip(panels, labels)):
            frame = np.asarray(frames[t % len(frames)])
            if frame.dtype != np.uint8:
                frame = np.rint(np.clip(frame,0,1)*255).astype(np.uint8)
            canvas.paste(Image.fromarray(frame), (j*side, 0))
            draw.text((j*side+3,side+3), label, fill='black')
            draw.text((j*side+3,side+19), f'color={row["color_id"]} shape={row["shape_id"]}', fill='black')
        return canvas
    stem = str(row['id'])
    path = output/f'{stem}.mp4'
    with av.open(str(path), mode='w') as container:
        stream = container.add_stream('libx264', rate=12)
        stream.width, stream.height = 4*side, height
        stream.pix_fmt = 'yuv420p'
        stream.options = {'crf':'18', 'preset':'fast'}
        for t in range(16):
            frame = av.VideoFrame.from_ndarray(np.asarray(compose(t)), format='rgb24')
            for packet in stream.encode(frame):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)
    sheet = Image.new('RGB', (4*side, 3*height), 'white')
    for j,t in enumerate((0,7,15)):
        sheet.paste(compose(t), (0,j*height))
    sheet.save(output/f'{stem}.png')
    return {'id': row['id'], 'pair_id': row['pair_id'], 'variant': row['variant'],
            'video': path.name, 'frame_sheet': f'{stem}.png', 'frames':16, 'fps':12,
            'cue': 'true external observation, never generated', 'panels': labels}


def calibrate(decoder, scorer):
    from data import load_rows, load_pair, render_chunk
    chosen, counts = [], defaultdict(int)
    for row in sorted(load_rows('val'), key=lambda r: str(r['id'])):
        attr = (int(row['color_id']),int(row['shape_id']))
        if counts[attr] < 3:
            chosen.append(row)
            counts[attr] += 1
    cases = []
    for index,row in enumerate(chosen):
        latent = load_pair(row['pair_id'])['target'][int(row['variant']):int(row['variant'])+1]
        decoded = decoder(latent)[0]
        truth, mask = render_chunk(row, kind='return', return_mask=True)
        cases.append({'id':row['id'], 'attr':row['attr'],
                      'decoded_truth':scorer.score(decoded,row,mask),
                      'renderer_truth':scorer.score(truth,row,mask)})
        if (index+1)%20==0 or index+1==len(chosen):
            print(f'calibration decoded/scored {index+1}/{len(chosen)} examples',flush=True)
    def summarize(key):
        return {metric:float(np.mean([c[key][metric] for c in cases])) for metric in METRICS} if cases else {}
    return {'n':len(cases), 'per_attribute_counts':{f'{c}/{s}':n for (c,s),n in sorted(counts.items())},
            'all_12_attributes_with_3_examples':len(counts)==12 and min(counts.values())>=3,
            'decoded_truth':summarize('decoded_truth'), 'renderer_truth':summarize('renderer_truth'),
            'cases':cases, 'role':'diagnostic only; no test-driven threshold fitting or hidden gate'}


@torch.inference_mode()
def main():
    from data import load_rows, render_chunk
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--pairs', type=int, default=60)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--preview-pairs', type=int, default=4)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    if args.pairs < 1 or args.preview_pairs < 0:
        parser.error('--pairs must be positive and --preview-pairs nonnegative')
    torch.set_num_threads(4)
    output = args.output or Path(WORK_ROOT)/'results'/f'seed{args.seed}'
    output.mkdir(parents=True, exist_ok=True)
    rows = balanced_subset(load_rows('test'), args.pairs)
    if not rows:
        raise ValueError('No complete test pairs')
    scorer = Scorer(rows[0])
    previews = preview_pair_ids(rows, args.preview_pairs)
    write_json(output/'manifest.json', {'rows':rows, 'noise_seed':2026, 'euler_steps':32,
        'guidance':1, 'preview_pair_ids':sorted(previews), 'score_frames':'last 8 return frames',
        'sampler':'pure shared pair noise; 32 Euler steps t=0 to 1; phase=1',
        'state':'N zero; C first true cue fixed throughout return; swap donor is paired variant',
        'decode':'independent chunk: duplicate first latent slice; decode; keep last16 frames',
        'checkpoint_step':6000, 'checkpoint_seed':args.seed})
    # Avoid spending GPU time if video export dependency is unavailable.
    if previews:
        import av
    started = time.perf_counter()
    latents, resources = generate_latents(rows,args)
    resources['model_phase_seconds'] = time.perf_counter()-started
    write_json(output/'resources.json',resources)
    decoder_started = time.perf_counter()
    gc.collect()
    if torch.device(args.device).type == 'cuda':
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(args.device)
    decoder = IndependentDecoder(args.device)
    calibration = calibrate(decoder,scorer)
    write_json(output/'calibration.json',calibration)
    cases, panels = [], defaultdict(dict)
    by_pair = defaultdict(dict)
    for row in rows:
        by_pair[str(row['pair_id'])][int(row['variant'])] = row
    decode_ms = 0.
    latents.sort(key=lambda x:str(x[0]['pair_id']))
    current_pair, decoded_pair = None, {}
    completed_pairs = 0
    for row, arm, condition, latent in latents:
        if str(row['pair_id']) != current_pair:
            if current_pair is not None:
                completed_pairs += 1
                if completed_pairs%10==0:
                    print(f'decoded/scored pairs {completed_pairs}/{len(rows)//2}',flush=True)
            current_pair = str(row['pair_id'])
            decoded_pair = {}
        if id(latent) not in decoded_pair:
            decoded, duration = timed(lambda:decoder(latent), args.device)
            decode_ms += duration
            decoded_pair[id(latent)] = decoded[0]
        frames = decoded_pair[id(latent)]
        _, mask = render_chunk(row,kind='return',return_mask=True)
        case = {'id':row['id'], 'pair_id':row['pair_id'], 'variant':row['variant'],
                'pair_change':row['pair_change'], 'attr':row['attr'], 'arm':arm,
                'condition':condition, **scorer.score(frames,row,mask)}
        if condition == 'swap':
            donor = by_pair[str(row['pair_id'])][1-int(row['variant'])]
            _, donor_mask = render_chunk(donor,kind='return',return_mask=True)
            donor_score = scorer.score(frames,donor,donor_mask)
            case.update(donor_id=donor['id'], donor_attr=donor['attr'],
                        donor_color_accuracy=donor_score['color_accuracy'],
                        donor_shape_accuracy=donor_score['shape_accuracy'],
                        donor_joint_accuracy=donor_score['joint_accuracy'])
        cases.append(case)
        if str(row['pair_id']) in previews and condition == 'normal':
            panels[str(row['id'])][arm] = np.rint(frames*255).astype(np.uint8)
    synchronize(args.device)
    resources['vae'] = {'model_resident':False,
                        'scope':'independent VAE loading, calibration and generated-chunk decoding; generator models unloaded',
                        'peak_allocated_bytes':None, 'peak_reserved_bytes':None}
    if torch.device(args.device).type == 'cuda':
        resources['vae'].update(peak_allocated_bytes=torch.cuda.max_memory_allocated(args.device),
                                peak_reserved_bytes=torch.cuda.max_memory_reserved(args.device))
    del latents, decoder
    print(f'decoded/scored pairs {len(rows)//2}/{len(rows)//2}',flush=True)
    resources.update(vae_generation_decode_ms=decode_ms,
                     vae_phase_seconds=time.perf_counter()-decoder_started)
    write_json(output/'resources.json',resources)
    metrics = aggregate(cases)
    intervention = {}
    normal = metrics.get('C/normal',{})
    zero = metrics.get('C/zero',{})
    for key in ('color_accuracy','shape_accuracy','joint_accuracy'):
        intervention[f'zero_drop_{key}'] = normal.get(key,0)-zero.get(key,0)
    swaps = [c for c in cases if c['condition']=='swap']
    for key in ('color_accuracy','shape_accuracy','joint_accuracy'):
        intervention[f'swap_donor_{key}'] = float(np.mean([c[f'donor_{key}'] for c in swaps])) if swaps else None
        intervention[f'swap_original_{key}'] = float(np.mean([c[key] for c in swaps])) if swaps else None
    intervention['paired_control_following'] = metrics.get('C/normal',{}).get('paired_control_following')
    summary = {'seed':args.seed, 'pairs':len(rows)//2, 'cases':len(cases),
               'metrics':metrics, 'interventions':intervention,
               'calibration_decoded_truth':calibration['decoded_truth'],
               'calibration_all_attributes':calibration['all_12_attributes_with_3_examples'],
               'thresholds':{'saturation':.22,'brightness':.35,'min_area_pixels':12,
                             'min_area_fraction':.0004,'max_area_fraction':.30,
                             'max_color_distance':.65,'min_aligned_shape_iou':.40},
               'shape_templates':'per-frame visible renderer silhouettes; bbox aligned; scene cached',
               'classification_mask':'largest connected saturated component; spatial IoU uses all saturated pixels',
               'paired_success_rule':'both variants joint accuracy > 0.5 over last8 frames; changed-attribute control following also requires > 0.5',
               'resources_file':'resources.json'}
    summary['deterministic_reuse'] = (
        'Per pair: N variants share one output; C zero variants share one output; '
        'C swap exactly reuses donor normal output. Four unique rollouts/decodes per pair; '
        'swap is a deterministic controlled response, not independent additional evidence.')
    write_json(output/'cases.json',cases)
    write_json(output/'summary.json',summary)
    with (output/'metrics.csv').open('w',encoding='utf-8',newline='') as handle:
        writer = csv.DictWriter(handle,fieldnames=['group',*next(iter(metrics.values())).keys()])
        writer.writeheader()
        writer.writerows({'group':key,**value} for key,value in metrics.items())
    samples = output/'samples'
    samples.mkdir(exist_ok=True)
    evidence = [export_preview(samples,row,panels[str(row['id'])]) for row in rows
                if str(row['pair_id']) in previews]
    write_json(samples/'manifest.json',evidence)
    print(f'Saved {len(cases)} cases and {len(evidence)} paired-variant videos to {output}',flush=True)


if __name__ == '__main__':
    main()
