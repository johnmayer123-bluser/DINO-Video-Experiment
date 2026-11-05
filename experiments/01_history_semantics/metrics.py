"""NumPy-only hierarchical means and paired source-group bootstrap."""

from collections import defaultdict

import numpy as np

NOISE_LEVELS = (.1, .3, .5, .7, .9)


def _aggregate(rows, errors, bootstraps, seed):
    arms = list(errors)
    arrays = {arm: np.asarray(value, dtype=np.float64) for arm, value in errors.items()}
    if not rows or any(v.shape != (len(rows), 5) or not np.isfinite(v).all() for v in arrays.values()):
        raise ValueError("Expected finite [window,5] errors aligned with a nonempty manifest")
    videos = defaultdict(list)
    for i, row in enumerate(rows):
        videos[(row["video"].split("/")[0], row["group"], row["video"])].append(
            np.stack([arrays[arm][i] for arm in arms]))
    groups = defaultdict(list)
    for (category, group, _), values in videos.items():
        groups[(category, group)].append(np.mean(values, axis=0))
    classes = defaultdict(list)
    for (category, _), values in sorted(groups.items()):
        classes[category].append(np.mean(values, axis=0))
    rng = np.random.default_rng(seed)
    point, samples = [], []
    for category in sorted(classes):
        values = np.stack(classes[category])
        # Draw a class's complete [replicate,group] block before the next class.
        # One draw is shared by every arm and noise level.
        indices = rng.integers(len(values), size=(bootstraps, len(values)))
        point.append(values.mean(0))
        samples.append(values[indices].mean(1))
    return arms, np.mean(point, axis=0), np.mean(samples, axis=0), {
        "windows": len(rows), "videos": len(videos), "source_groups": len(groups), "classes": len(classes)}


def _pair(arms, point, samples, a, b):
    ia, ib = arms.index(a), arms.index(b)
    slices = {}
    for name, cols in {"all": tuple(range(5)), **{f"t_{t:.1f}": (i,) for i,t in enumerate(NOISE_LEVELS)}}.items():
        ma, mb = float(point[ia, list(cols)].mean()), float(point[ib, list(cols)].mean())
        ba, bb = samples[:, ia, cols].mean(1), samples[:, ib, cols].mean(1)
        slices[name] = {"mse_a": ma, "mse_b": mb, "relative_improvement": 1-mb/ma,
                        "ci95": {key: np.quantile(value, [.025,.975]).tolist() for key,value in
                                 {"mse_a":ba, "mse_b":bb, "relative_improvement":1-bb/ba}.items()}}
    return {"primary": slices["all"], "slices": slices, "noise_levels": list(NOISE_LEVELS)}


def summarize_pair(rows, a, b, bootstraps=5000, seed=2026):
    """Average windows -> videos -> source groups -> classes; pair all resamples."""
    arms, point, samples, counts = _aggregate(rows, {"a":a,"b":b}, bootstraps, seed)
    return {**_pair(arms,point,samples,"a","b"), "class_count":counts["classes"],
            "group_count":counts["source_groups"]}


def summarize(rows, errors, bootstraps=5000, seed=2026):
    """Summarize N/S/SD/L errors after averaging each window's noise repeats."""
    arms, point, samples, counts = _aggregate(rows, errors, bootstraps, seed)
    pairs = {"N_vs_SD": ("N","SD"), "S_vs_SD": ("S","SD"),
             "N_vs_S": ("N","S"), "N_vs_L": ("N","L")}
    return {"mse": {arm:float(point[i].mean()) for i,arm in enumerate(arms)},
            "by_t": {arm:point[i].tolist() for i,arm in enumerate(arms)},
            "pairs": {key:_pair(arms,point,samples,a,b) for key,(a,b) in pairs.items()}, "counts":counts}
