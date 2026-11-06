"""Reuse Q6 scenes and paired tensors; independently encode full blank chunks.

python data.py --index-only
python data.py --device cuda --batch-size 8
No cue or return chunk is encoded, and the Q6 cache is read only.
"""
import argparse
from functools import lru_cache
import json
import time

from common import Q6_ROOT, REPO_ROOT, WORK_ROOT, load_source, write_json

q6 = load_source("q7_base_data", Q6_ROOT / "data.py")
q6.CACHE = Q6_ROOT / "cache"
COLORS, SHAPES, ATTRIBUTES = q6.COLORS, q6.SHAPES, q6.ATTRIBUTES
FRAMES, FPS, RESOLUTION = q6.FRAMES, q6.FPS, q6.RESOLUTION
LATENT_SHAPE = (16, 4, 24, 24)
render_chunk, load_rows = q6.render_chunk, q6.load_rows
CACHE = WORK_ROOT / "cache"


@lru_cache(maxsize=8)
def load_pair(pair_id):
    """CPU tensors: Q6 boundary/cue/target/mask plus full scene blank hidden."""
    import torch
    payload = dict(q6.load_pair(pair_id))
    hidden = torch.load(CACHE / "blanks" / f"{pair_id}.pt",
                        map_location="cpu", weights_only=True)
    if not isinstance(hidden, torch.Tensor) or tuple(hidden.shape) != LATENT_SHAPE:
        raise ValueError(f"Invalid full blank latent for {pair_id}")
    payload["hidden"] = hidden
    return payload


def _manifest_pairs():
    rows = json.loads((q6.CACHE / "manifest.json").read_text(encoding="utf-8"))
    groups = {}
    for row in rows:
        groups.setdefault(row["pair_id"], []).append(row)
    for pair_id, pair in groups.items():
        if len(pair) != 2 or {r["variant"] for r in pair} != {0, 1}:
            raise ValueError(f"Invalid Q6 pair: {pair_id}")
        if pair[0]["scene"] != pair[1]["scene"]:
            raise ValueError(f"Q6 variants do not share a scene: {pair_id}")
        pair.sort(key=lambda row: row["variant"])
    return rows, list(groups.values())


def prepare(args):
    """Build only Q7 full blank latents, batching distinct scenes together."""
    started = time.perf_counter()
    if args.batch_size < 1 or (args.limit_pairs is not None and args.limit_pairs < 1):
        raise ValueError("batch-size and limit-pairs must be positive")
    rows, pairs = _manifest_pairs()
    blanks = CACHE / "blanks"
    blanks.mkdir(parents=True, exist_ok=True)
    chosen = pairs if args.limit_pairs is None else pairs[:args.limit_pairs]
    missing_q6 = [p[0]["pair_id"] for p in chosen
                  if not (q6.CACHE / "tensors" / f"{p[0]['pair_id']}.pt").exists()]
    if missing_q6 and not args.index_only:
        raise FileNotFoundError(f"Missing Q6 paired tensors: {missing_q6[:5]}")
    write_json(CACHE / "manifest.json", rows)
    source_metadata_path = q6.CACHE / "metadata.json"
    metadata = json.loads(source_metadata_path.read_text(encoding="utf-8"))
    metadata.update({
        "version": 1,
        "experiment": "recursive_closed_loop_v1",
        "source_experiment": "keep_condition_v1",
        "source_manifest": str(q6.CACHE / "manifest.json"),
        "source_pair_tensors": str(q6.CACHE / "tensors"),
        "blank_tensor_directory": str(blanks),
        "blank_latent_shape": list(LATENT_SHAPE),
        "blank_encoding": "independent scene blank: [repeat first,16 frames], frozen local Wan VAE posterior.mode; normalize; discard first latent",
        "normalization": "(posterior.mode - config.latents_mean) / config.latents_std",
        "storage_dtype": "float16",
        "boundary": "reuse exact Q6 final retained slice of independent scene blank; never repeat boundary to synthesize hidden",
        "tensor_fields": {"boundary": [16, 1, 24, 24], "cue": [2, 16, 4, 24, 24],
                          "target": [2, 16, 4, 24, 24], "mask": [2, 1, 4, 24, 24],
                          "hidden": list(LATENT_SHAPE)},
    })
    write_json(CACHE / "metadata.json", metadata)
    pending = [p for p in chosen if not (blanks / f"{p[0]['pair_id']}.pt").exists()]
    encoded_count = 0
    if pending and not args.index_only:
        import torch
        from diffusers import AutoencoderKLWan
        torch.set_num_threads(4)
        device = torch.device(args.device)
        model = AutoencoderKLWan.from_pretrained(
            REPO_ROOT / "models" / "wan-vae", torch_dtype=torch.float32,
            local_files_only=True).to(device).eval().requires_grad_(False)
        mean = torch.tensor(model.config.latents_mean, device=device).view(1, 16, 1, 1, 1)
        std = torch.tensor(model.config.latents_std, device=device).view(1, 16, 1, 1, 1)
        metadata["vae_config_mean"] = list(model.config.latents_mean)
        metadata["vae_config_std"] = list(model.config.latents_std)
        write_json(CACHE / "metadata.json", metadata)
        for start in range(0, len(pending), args.batch_size):
            batch = pending[start:start + args.batch_size]
            chunks = [render_chunk(pair[0], "blank") for pair in batch]
            encoded = q6.encode_chunks(model, chunks, mean, std, device,
                                       batch_size=args.batch_size)
            for pair, latent in zip(batch, encoded):
                pair_id = pair[0]["pair_id"]
                # q6.encode_chunks returns unbound views: own storage per file.
                latent = latent.to(device="cpu", dtype=torch.float16).clone().contiguous()
                if tuple(latent.shape) != LATENT_SHAPE or not torch.isfinite(latent).all():
                    raise ValueError(f"Invalid encoded blank for {pair_id}")
                path = blanks / f"{pair_id}.pt"
                temporary = path.with_suffix(".pt.tmp")
                torch.save(latent, temporary)
                temporary.replace(path)
                encoded_count += 1
                if encoded_count % 25 == 0 or encoded_count == len(pending):
                    print(f"blank cache {encoded_count}/{len(pending)} {pair_id}", flush=True)
            del encoded, chunks
        del model
    present = sum((blanks / f"{p[0]['pair_id']}.pt").exists() for p in pairs)
    summary = {"pairs": len(pairs), "variant_rows": len(rows),
               "cached_pairs": present, "complete": present == len(pairs),
               "selected_pairs": len(chosen), "encoded_this_run": encoded_count,
               "index_only": args.index_only,
               "elapsed_seconds": time.perf_counter() - started}
    write_json(CACHE / "summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--limit-pairs", type=int)
    parser.add_argument("--index-only", action="store_true")
    args = parser.parse_args()
    if args.batch_size < 1 or (args.limit_pairs is not None and args.limit_pairs < 1):
        parser.error("batch-size and limit-pairs must be positive")
    prepare(args)


if __name__ == "__main__":
    main()
