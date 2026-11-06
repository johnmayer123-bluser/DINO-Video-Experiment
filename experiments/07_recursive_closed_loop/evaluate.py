"""Q7 paired causal closed-loop evaluation (single seed 42).

python evaluate.py --pairs 60 --seed 42 --batch-pairs 8 --device cuda
Batch-one latency is measured separately by resources.py, never inferred here.
"""
import argparse
from collections import defaultdict
import csv
import gc
from pathlib import Path
import time

import numpy as np
from PIL import Image, ImageDraw
import torch

from common import Q6_ROOT, WORK_ROOT, load_source, write_json

q6 = load_source('recursive_q6_evaluate', Q6_ROOT / 'evaluate.py')
Scorer, IndependentDecoder = q6.Scorer, q6.IndependentDecoder
balanced_subset, timed = q6.balanced_subset, q6.timed
unique_storage_kib, METRICS = q6.unique_storage_kib, q6.METRICS


def chunk_noise(pair_id, chunk=None):
    # Counterfactual variants, arms, and history modes use identical noise.
    # Return branches also share noise across every delay and intervention.
    return q6.pair_noise(f'{pair_id}|return' if chunk is None else f'{pair_id}|blank{chunk}')


@torch.inference_mode()
def sample_chunk(model, boundary, state, noise, phase=1):
    from model import precision
    x = noise.to(boundary.device).float().clone()
    for step in range(32):
        t = torch.full((len(x),), step / 32, device=x.device)
        with precision(x.device):
            velocity = model.predict(boundary, x, t, state, phase=phase)
        x.add_(velocity.float(), alpha=1 / 32)
    return x


def preview_pair_ids(rows, count=2):
    chosen = []
    for change in ('color', 'shape'):
        candidates = sorted({str(r['pair_id']) for r in rows if r['pair_change'] == change})
        if candidates:
            chosen.append(candidates[0])
    return set(chosen[:count])


@torch.inference_mode()
def generate_latents(rows, args):
    from data import load_pair
    from model import ARMS, load_model, precision
    records, previews, diagnostics = [], defaultdict(dict), []
    preview_ids = preview_pair_ids(rows, args.preview_pairs)
    leak_ids = {str(row['pair_id']) for row in rows[:4]}
    pair_groups = [rows[i:i+2] for i in range(0, len(rows), 2)]
    max_delay = max(args.delays)
    for arm in ARMS:
        gc.collect()
        if torch.device(args.device).type == 'cuda':
            torch.cuda.empty_cache()
        model = load_model(arm, device=args.device, step=6000, seed=args.seed)
        for start in range(0, len(pair_groups), args.batch_pairs):
            groups = pair_groups[start:start+args.batch_pairs]
            batch_rows = [r for group in groups for r in group]
            items = [load_pair(group[0]['pair_id']) for group in groups]
            cue = torch.cat([item['cue'] for item in items]).to(args.device).float()
            hidden = torch.stack([item['hidden'] for item in items]).repeat_interleave(2, 0).to(args.device).float()
            canonical = torch.stack([item['boundary'] for item in items]).repeat_interleave(2, 0).to(args.device).half()
            noise = torch.cat([chunk_noise(group[0]['pair_id']).expand(2,-1,-1,-1,-1) for group in groups])
            with precision(torch.device(args.device)):
                initial = model.observe(cue)
            # Independent truth/closed states; each chain advances only once.
            states = {'truth': initial, 'closed': initial.clone()}
            boundary = cue[:, :, -1:].half().clone()
            truth_reuse = None
            for delay in range(1, max_delay+1):
                blank_noise = torch.cat([chunk_noise(group[0]['pair_id'], delay).expand(2,-1,-1,-1,-1) for group in groups])
                blank = sample_chunk(model, boundary, states['closed'], blank_noise, phase=0)
                with precision(torch.device(args.device)):
                    states['closed'] = model.observe(blank, states['closed'])
                    states['truth'] = model.observe(hidden, states['truth'])
                # A clone owns just the final slice, not all four latent slices.
                boundary = blank[:, :, -1:].half().clone()
                for i, row in enumerate(batch_rows):
                    pair_id, row_id = str(row['pair_id']), str(row['id'])
                    if pair_id in preview_ids:
                        previews[row_id].setdefault(arm, []).append(blank[i:i+1].half().cpu().clone())
                    if delay in args.delays:
                        diagnostics.append({'id':row['id'], 'pair_id':row['pair_id'], 'variant':row['variant'],
                            'arm':arm, 'delay':delay, 'hidden_latent_mse':float((blank[i].float()-hidden[i]).square().mean()),
                            'state_norm':float(states['closed'][i].float().norm()),
                            'truth_state_norm':float(states['truth'][i].float().norm())})
                        if pair_id in leak_ids:
                            # Keep only selected endpoints, not full non-preview histories.
                            previews[row_id].setdefault(f'{arm}_leak', {})[delay] = blank[i:i+1].half().cpu().clone()
                if delay not in args.delays:
                    continue
                for history, branch_boundary in (('truth',canonical), ('closed',boundary)):
                    # F retains s0 and L always re-encodes the identical canonical blank.
                    if history == 'truth' and arm in ('F','L') and truth_reuse is not None:
                        returned = truth_reuse
                    else:
                        returned = sample_chunk(model, branch_boundary, states[history], noise).half().cpu()
                        if history == 'truth' and arm in ('F','L'):
                            truth_reuse = returned
                    for i,row in enumerate(batch_rows):
                        records.append({'row':row, 'arm':arm, 'history':history, 'delay':delay,
                                        'condition':'normal', 'latent':returned[i:i+1]})
                if arm == 'R' and delay in (1,32):
                    for condition,state in (('clamp_boundary',states['closed']),
                                            ('clamp_boundary_zero',torch.zeros_like(states['closed']))):
                        returned = sample_chunk(model, canonical, state, noise).half().cpu()
                        for i,row in enumerate(batch_rows):
                            records.append({'row':row,'arm':arm,'history':'closed','delay':delay,
                                            'condition':condition,'latent':returned[i:i+1]})
            # Loop aliases can otherwise retain device tensors past their batch.
            del cue, hidden, canonical, states, boundary, blank, initial, returned
            del branch_boundary
            if arm == 'R' and any(d in args.delays for d in (1,32)):
                del state
            print(f'generated {arm} pairs {min(start+args.batch_pairs,len(pair_groups))}/{len(pair_groups)}',flush=True)
        del model
    gc.collect()
    if torch.device(args.device).type == 'cuda':
        torch.cuda.empty_cache()
    return records, dict(previews), diagnostics


def aggregate(cases):
    groups = defaultdict(list)
    for case in cases:
        groups[f'{case["arm"]}/{case["history"]}/d{case["delay"]}/{case["condition"]}'].append(case)
    result = {}
    for key,members in groups.items():
        pairs = defaultdict(list)
        for case in members:
            pairs[str(case['pair_id'])].append(case)
        complete = [p for p in pairs.values() if len(p)==2]
        result[key] = {'n':len(members), **{m:float(np.mean([c[m] for c in members])) for m in METRICS},
            'paired_both_correct':float(np.mean([all(c['joint_accuracy']>.5 for c in p) for p in complete])) if complete else None,
            'paired_control_following':float(np.mean([all(c[f'{c["pair_change"]}_accuracy']>.5 for c in p) for p in complete])) if complete else None}
        for kind in ('color','shape'):
            coverage = sum(c[f'{kind}_classifiable'] for c in members)
            result[key][f'{kind}_accuracy_when_classifiable'] = sum(c[f'{kind}_accuracy'] for c in members)/coverage if coverage else None
    return result


def export_preview(output, row, delay, blanks, returns):
    """Full cue + delay blank chunks + return, 2 history rows x 3 arms."""
    import av
    from data import render_chunk
    cue, blank = render_chunk(row,'cue'), render_chunk(row,'blank')
    side, label_h = cue.shape[1], 44
    cell_h = side+label_h
    arms = ('F','L','R')
    def compose(t):
        chunk, within = divmod(t,16)
        canvas = Image.new('RGB',(side*3,cell_h*2),'white')
        draw = ImageDraw.Draw(canvas)
        for h,history in enumerate(('truth','closed')):
            for a,arm in enumerate(arms):
                if chunk == 0:
                    frame,kind = cue[within], 'TRUE cue'
                elif chunk <= delay:
                    frame = blank[within] if history=='truth' else blanks[arm][chunk-1][within]
                    kind = 'TRUE blank' if history=='truth' else 'GENERATED blank'
                else:
                    frame,kind = returns[(arm,history,delay)][within], 'GENERATED return'
                if frame.dtype != np.uint8:
                    frame = np.rint(np.clip(frame,0,1)*255).astype(np.uint8)
                left,top = a*side,h*cell_h
                canvas.paste(Image.fromarray(frame),(left,top))
                draw.text((left+3,top+side+2),f'{arm} {history} | {kind} {chunk}/{delay+1}',fill='black')
                draw.text((left+3,top+side+18),f'color={row["color_id"]} shape={row["shape_id"]}; cue observed',fill='black')
                if history=='closed' and 0<chunk<=delay:
                    area = float(q6.visible_mask(frame).mean())
                    draw.text((left+3,top+side+30),f'saturated area={area:.4f}',fill='black')
        return canvas
    stem = f'{row["id"]}_d{delay}'
    frames = (delay+2)*16
    with av.open(str(output/f'{stem}.mp4'),'w') as container:
        stream = container.add_stream('libx264',rate=12)
        stream.width,stream.height = side*3,cell_h*2
        stream.pix_fmt = 'yuv420p'
        stream.options = {'crf':'18','preset':'fast'}
        for t in range(frames):
            frame = av.VideoFrame.from_ndarray(np.asarray(compose(t)),format='rgb24')
            for packet in stream.encode(frame):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)
    sheet = Image.new('RGB',(side*3,cell_h*8),'white')
    for i,t in enumerate((0,16,delay*16+15,frames-1)):
        sheet.paste(compose(t),(0,i*cell_h*2))
    sheet.save(output/f'{stem}.png')
    return {'id':row['id'],'pair_id':row['pair_id'],'variant':row['variant'],'delay':delay,
            'video':f'{stem}.mp4','frame_sheet':f'{stem}.png','frames':frames,'fps':12,
            'layout':'rows truth/closed; columns F/L/R; cue true; closed blank generated; return generated'}


@torch.inference_mode()
def main():
    from data import load_rows, render_chunk
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--pairs',type=int,default=60)
    parser.add_argument('--seed',type=int,default=42,choices=[42])
    parser.add_argument('--device',default='cuda')
    parser.add_argument('--delays',type=int,nargs='+',default=[1,4,8,16,32])
    parser.add_argument('--batch-pairs',type=int,default=8)
    parser.add_argument('--preview-pairs',type=int,default=2)
    parser.add_argument('--output',type=Path)
    args = parser.parse_args()
    if args.pairs<1 or args.batch_pairs<1 or not args.delays or min(args.delays)<1 or max(args.delays)>32 or not 0<=args.preview_pairs<=2:
        parser.error('positive pairs/batch-pairs, delays 1..32 and preview-pairs 0..2 required')
    args.delays = sorted(set(args.delays))
    torch.set_num_threads(4)
    output = args.output or WORK_ROOT/'results'/f'seed{args.seed}'
    output.mkdir(parents=True,exist_ok=True)
    rows = balanced_subset(load_rows('test'),args.pairs)
    if not rows:
        raise ValueError('No complete test pairs')
    if args.preview_pairs:
        import av
    write_json(output/'manifest.json',{'rows':rows,'seed':42,'checkpoint_step':6000,'delays':args.delays,
        'batch_pairs':args.batch_pairs,'sample_steps':32,'noise':'pair-shared per blank index; return shared across delays, histories and interventions',
        'history':'s0=observe(true cue); blank_j=sample(s_j-1,b_j-1,phase0); s_j=observe(blank_j,s_j-1); b_j=blank_j final slice; return is a non-feedback branch phase1',
        'truth':'observe full canonical hidden blank at every step; canonical blank boundary for return',
        'preview_pair_ids':sorted(preview_pair_ids(rows,args.preview_pairs)),
        'resource_note':'quality batches are throughput only; run resources.py for independent batch1 measurements'})
    started = time.perf_counter()
    records,previews,diagnostics = generate_latents(rows,args)
    write_json(output/'rollout_diagnostics.json',diagnostics)
    resources = {'quality_model_phase_seconds':time.perf_counter()-started,'quality_batch_pairs':args.batch_pairs,
                 'batch1_latency_measured':False,'batch1_resources_command':'python resources.py --device cuda',
                 'persistent_condition_kib':26,'vae':{}}
    decoder_started = time.perf_counter()
    if torch.device(args.device).type=='cuda':
        torch.cuda.reset_peak_memory_stats(args.device)
    decoder = IndependentDecoder(args.device)
    scorer = Scorer(rows[0])
    calibration = q6.calibrate(decoder,scorer)
    write_json(output/'calibration.json',calibration)
    cases,panels,leakage = [],defaultdict(dict),[]
    # Bounded decoder memoization per pair; equivalent truth delays share storage.
    records.sort(key=lambda r:str(r['row']['pair_id']))
    current_pair,decoded_cache,completed_pairs = None,{},0
    for record in records:
        row,latent = record['row'],record['latent']
        pair_id = str(row['pair_id'])
        if pair_id != current_pair:
            if current_pair is not None:
                completed_pairs += 1
                if completed_pairs % 10 == 0:
                    print(f'decoded/scored pairs {completed_pairs}/{len(rows)//2}',flush=True)
            current_pair,decoded_cache = pair_id,{}
        cache_key = (latent.untyped_storage().data_ptr(),latent.storage_offset())
        if cache_key not in decoded_cache:
            decoded_cache[cache_key] = decoder(latent)[0]
        frames = decoded_cache[cache_key]
        _,mask = render_chunk(row,'return',return_mask=True)
        case = {k:record[k] for k in ('arm','history','delay','condition')}
        case.update({k:row[k] for k in ('id','pair_id','variant','pair_change','attr')})
        case.update(scorer.score(frames,row,mask))
        cases.append(case)
        if str(row['id']) in previews and record['condition']=='normal' and record['delay'] in (4,32):
            panels[str(row['id'])][(record['arm'],record['history'],record['delay'])] = np.rint(frames*255).astype(np.uint8)
    print(f'decoded/scored pairs {len(rows)//2}/{len(rows)//2}',flush=True)
    decoded_blanks = defaultdict(dict)
    for preview_index,(row_id,arms) in enumerate(previews.items(),1):
        for arm,chunks in arms.items():
            if arm.endswith('_leak'):
                for delay,latent in chunks.items():
                    frames = decoder(latent)[0]
                    area = q6.visible_mask(frames).reshape(len(frames),-1).mean(1)
                    threshold = max(12,round(frames.shape[1]*frames.shape[2]*.0004))
                    leakage.append({'id':row_id,'arm':arm.split('_')[0],'delay':delay,
                        'saturated_area_fraction':float(area.mean()),'last8_saturated_area_fraction':float(area[-8:].mean()),
                        'visible_presence_fraction':float(np.mean(area*frames.shape[1]*frames.shape[2]>=threshold))})
            else:
                decoded_blanks[row_id][arm] = [np.rint(decoder(chunk)[0]*255).astype(np.uint8) for chunk in chunks]
        print(f'blank preview decoded row {row_id} ({preview_index}/{len(previews)})',flush=True)
    resources['vae'].update(generator_resident=False,seconds=time.perf_counter()-decoder_started,
        peak_allocated_bytes=torch.cuda.max_memory_allocated(args.device) if torch.device(args.device).type=='cuda' else None,
        peak_reserved_bytes=torch.cuda.max_memory_reserved(args.device) if torch.device(args.device).type=='cuda' else None)
    del decoder,records,previews
    metrics = aggregate(cases)
    write_json(output/'cases.json',cases)
    write_json(output/'blank_leakage.json',leakage)
    write_json(output/'resources.json',resources)
    write_json(output/'summary.json',{'seed':42,'pairs':len(rows)//2,'cases':len(cases),'metrics':metrics,
        'calibration_decoded_truth':calibration['decoded_truth'],
        'paired_success_rule':'both variant accuracies strictly >0.5 on last8 frames',
        'interpretation':'Single seed diagnostic; blank presence/area and clamp interventions test visible-boundary leakage.',
        'resources_file':'resources.json','blank_leakage_file':'blank_leakage.json'})
    with (output/'metrics.csv').open('w',encoding='utf-8',newline='') as handle:
        writer = csv.DictWriter(handle,fieldnames=['group',*next(iter(metrics.values())).keys()])
        writer.writeheader()
        writer.writerows({'group':key,**value} for key,value in metrics.items())
    samples = output/'samples'
    samples.mkdir(exist_ok=True)
    evidence = [export_preview(samples,row,delay,decoded_blanks[str(row['id'])],panels[str(row['id'])])
                for row in rows if str(row['pair_id']) in preview_pair_ids(rows,args.preview_pairs)
                for delay in (4,32) if delay in args.delays]
    write_json(samples/'manifest.json',evidence)
    print(f'Saved {len(cases)} cases, {len(evidence)} full causal videos to {output}',flush=True)


if __name__=='__main__':
    main()
