"""Fixed paired six-panel video evidence from independently decoded chunks.

Each panel shows external cue observations, hidden history, then a generated
return. Rows are truth-history/closed-loop; columns are recent/first/memory.
Only one generator is resident alongside the VAE. Temporary CPU panel files
keep rendering memory bounded and are removed after MP4/contact-sheet export.
"""
import argparse
import gc
import json
from pathlib import Path
import tempfile

import numpy as np
from PIL import Image, ImageDraw
import torch

from data import ATTRIBUTES, episode_latents
from model import ROOT, ARMS, load_model, precision
from evaluate import IndependentDecoder, cpu_noise, sample_chunk, write_json

HISTORIES = ('truth_history', 'closed_loop')
FPS, SIDE, PANEL_HEIGHT = 12, 192, 224


def fixed_rows(root):
    """Use manifest order only, before reading any model output."""
    path = Path(root)/'results/delayed_memory/test/evaluation_manifest.json'
    manifest = json.loads(path.read_text(encoding='utf-8'))
    selected = []
    for mode in ('keep', 'update'):
        first = next((r for r in manifest if r['mode'] == mode), None)
        if first is None:
            raise ValueError(f'Evaluation manifest has no {mode} pair')
        members = sorted([r for r in manifest if r['pair_id'] == first['pair_id']],
                         key=lambda r: r['variant'])
        if len(members) != 2 or [r['variant'] for r in members] != [0, 1]:
            raise ValueError('Evidence requires complete two-variant pairs')
        selected.extend(members)
    return selected


def uint8_frames(rgb):
    return np.rint(np.clip(rgb, 0, 1)*255).astype(np.uint8)


def label_attr(value):
    colors = ('red', 'green', 'blue', 'yellow')
    value = int(value)
    return f'{colors[value//3]} {ATTRIBUTES[value]["shape"]}'


def phase_caption(index, cue_count, delay):
    chunk = index//16
    if chunk < cue_count:
        return f'cue {chunk+1}/{cue_count}'
    if chunk < cue_count+delay:
        return f'hidden {chunk-cue_count+1}/{delay}'
    return 'return'


def panel_image(frame, arm, history, phase, expected):
    panel = Image.new('RGB', (SIDE, PANEL_HEIGHT), 'white')
    panel.paste(Image.fromarray(frame), (0, 0))
    draw = ImageDraw.Draw(panel)
    name = 'truth' if history == 'truth_history' else 'closed'
    draw.text((3, SIDE+1), f'{arm} / {name} | {phase}', fill='black')
    draw.text((3, SIDE+15), f'return target: {expected}', fill='black')
    return panel


def compose_frame(panels, index, cue_count, delay, expected):
    canvas = Image.new('RGB', (3*SIDE, 2*PANEL_HEIGHT), 'white')
    phase = phase_caption(index, cue_count, delay)
    for row, history in enumerate(HISTORIES):
        for col, arm in enumerate(ARMS):
            canvas.paste(panel_image(panels[(arm, history)][index], arm, history,
                                     phase, expected), (col*SIDE, row*PANEL_HEIGHT))
    return canvas


def export_case(output, row, delay, panels):
    """Encode H.264 at 12fps; four contact-sheet columns show fixed times."""
    import av
    cue_count = 1 if row['mode'] == 'keep' else 2
    total = 16*(cue_count+delay+1)
    stem = f'{row["id"]}_d{delay}'
    video_path, sheet_path = output/f'{stem}.mp4', output/f'{stem}_contact.png'
    expected = label_attr(row['final_attr'])
    with av.open(str(video_path), mode='w') as container:
        stream = container.add_stream('libx264', rate=FPS)
        stream.width, stream.height = 3*SIDE, 2*PANEL_HEIGHT
        stream.pix_fmt = 'yuv420p'
        stream.options = {'crf': '18', 'preset': 'medium'}
        for index in range(total):
            image = compose_frame(panels, index, cue_count, delay, expected)
            frame = av.VideoFrame.from_ndarray(np.asarray(image), format='rgb24')
            for packet in stream.encode(frame):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)
    positions = [16*cue_count-1, 16*(cue_count+delay)-1,
                 16*(cue_count+delay)+7, total-1]
    # Six panel rows, four time columns, retain original 192px image resolution.
    sheet = Image.new('RGB', (4*SIDE, 6*PANEL_HEIGHT+24), 'white')
    draw = ImageDraw.Draw(sheet)
    titles = ('last cue', 'last hidden', 'return middle', 'return end')
    for col, title in enumerate(titles):
        draw.text((col*SIDE+3, 5), title, fill='black')
    for panel_row, (history, arm) in enumerate((h, a) for h in HISTORIES for a in ARMS):
        for col, index in enumerate(positions):
            image = panel_image(panels[(arm, history)][index], arm, history,
                                phase_caption(index, cue_count, delay), expected)
            sheet.paste(image, (col*SIDE, 24+panel_row*PANEL_HEIGHT))
    sheet.save(sheet_path)
    return {'id': row['id'], 'pair_id': row['pair_id'], 'variant': row['variant'],
            'mode': row['mode'], 'delay': delay, 'initial_attr': row['initial_attr'],
            'expected_attr': row['final_attr'], 'expected_label': expected,
            'frames': total, 'fps': FPS, 'video': video_path.name,
            'contact_sheet': sheet_path.name, 'contact_frame_indices': positions}


@torch.inference_mode()
def generate(args):
    import av  # Fail before expensive generation if the pinned video dependency is missing.
    torch.set_num_threads(4)
    rows = fixed_rows(args.root)
    output = args.root/'experiments/06_delayed_memory/results/samples'
    output.mkdir(parents=True, exist_ok=True)
    # Save the fixed selection even if later generation is interrupted.
    metadata = {'seed': args.seed, 'delays': args.delays, 'ids': [r['id'] for r in rows],
                'selection': 'first manifest pair per keep/update mode; both variants; no outcome selection',
                'source_manifest': 'results/delayed_memory/test/evaluation_manifest.json',
                'arms_columns': list(ARMS), 'history_rows': list(HISTORIES),
                'euler_steps': 32, 'fps': FPS, 'noise_seed': 2026,
                'noise': 'evaluate.cpu_noise: SHA256(2026,pair_id,delay,chunk); shared arms/histories/variants',
                'phase': 'externally known schedule; hidden=0, return=1; no attribute input',
                'observations': 'one cue for keep, two external cues for update; no cue generation',
                'truth_history_display': 'independent VAE reconstruction of real hidden chunk',
                'closed_loop_display': 'independent VAE decode of each generated hidden chunk',
                'decode': 'duplicate each chunk first latent as synthetic anchor, decode independently, last16 RGB frames; no GT anchor',
                'checkpoints': {}, 'cases': []}
    write_json(output/'manifest.json', metadata)
    decoder = IndependentDecoder(args.root, 'cuda')
    cache = {r['id']: episode_latents(args.root, r) for r in rows}
    cue_rgb, hidden_rgb = {}, {}
    for row in rows:
        item = cache[row['id']]
        cue_rgb[row['id']] = [uint8_frames(decoder(cue[None])[0]) for cue in item['cues']]
        if row['pair_id'] not in hidden_rgb:
            hidden_rgb[row['pair_id']] = uint8_frames(decoder(item['hidden'][None])[0])
    with tempfile.TemporaryDirectory(prefix='panels_', dir=output) as temporary:
        temporary = Path(temporary)
        for arm in ARMS:
            gc.collect()
            torch.cuda.empty_cache()
            model = load_model(args.root, args.seed, arm, 'cuda')
            checkpoint = torch.load(args.root/'runs'/f'delayed_memory_seed{args.seed}'/'final.pt',
                                    map_location='cpu', weights_only=False)
            metadata['checkpoints'][arm] = {'step': checkpoint['step'], 'config': checkpoint['config']}
            del checkpoint
            for row in rows:
                item = cache[row['id']]
                state = None
                with precision(torch.device('cuda')):
                    for cue in item['cues']:
                        observed = cue[None].cuda().float()
                        state = model.observe(observed, state)
                    boundary = observed[:, :, -1:].half().clone()
                del observed
                initial_state, initial_boundary = state.clone(), boundary.clone()
                for delay in args.delays:
                    for history in HISTORIES:
                        state, boundary = initial_state.clone(), initial_boundary.clone()
                        frames = list(cue_rgb[row['id']])
                        for chunk in range(delay):
                            if history == 'truth_history':
                                current = item['hidden'][None].cuda().float()
                                frames.append(hidden_rgb[row['pair_id']])
                            else:
                                current = sample_chunk(model, boundary, state,
                                                       cpu_noise([row['id']], delay, chunk), phase=0)
                                frames.append(uint8_frames(decoder(current)[0]))
                            with precision(torch.device('cuda')):
                                state = model.observe(current, state)
                            boundary = current[:, :, -1:].half().clone()
                            del current
                        returned = sample_chunk(model, boundary, state,
                                                cpu_noise([row['id']], delay, delay), phase=1)
                        frames.append(uint8_frames(decoder(returned)[0]))
                        np.save(temporary/f'{row["id"]}_d{delay}_{arm}_{history}.npy', np.concatenate(frames))
                        del returned, frames
                        print(f'video panels: {arm} {row["id"]} delay={delay} {history}', flush=True)
                del state, boundary, initial_state, initial_boundary
            del model
            gc.collect()
            torch.cuda.empty_cache()
            write_json(output/'manifest.json', metadata)
        for row in rows:
            for delay in args.delays:
                panels = {(arm, history): np.load(temporary/f'{row["id"]}_d{delay}_{arm}_{history}.npy', mmap_mode='r')
                          for arm in ARMS for history in HISTORIES}
                metadata['cases'].append(export_case(output, row, delay, panels))
                for panel in panels.values():
                    panel._mmap.close()
                del panels
                write_json(output/'manifest.json', metadata)
    print(f'Saved {len(metadata["cases"])} six-panel videos and contact sheets to {output}', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=ROOT)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--delays', nargs='+', type=int, default=[3, 6])
    args = parser.parse_args()
    if not args.delays or any(d not in (3, 6) for d in args.delays):
        parser.error('--delays must contain 3 and/or 6')
    args.delays = list(dict.fromkeys(args.delays))
    generate(args)


if __name__ == '__main__':
    main()
