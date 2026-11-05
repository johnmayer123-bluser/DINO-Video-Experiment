"""Paired validation summaries for the latent denoising A/B experiment."""

from collections import defaultdict

import numpy as np


NOISE_LEVELS = (0.1, 0.3, 0.5, 0.7, 0.9)
SLICES = {
    "t_0.1": (0,),
    "t_0.3": (1,),
    "t_0.5": (2,),
    "t_0.7": (3,),
    "t_0.9": (4,),
    "all": (0, 1, 2, 3, 4),
    "high_noise": (0, 1),
    "mid_noise": (2,),
    "low_noise": (3, 4),
}


def _group_errors(rows, errors_a, errors_b):
    """Average windows into videos, then videos into source groups."""
    videos = defaultdict(lambda: [[], []])
    for row, value_a, value_b in zip(rows, errors_a, errors_b):
        key = (row["video"].split("/")[0], row["group"], row["video"])
        videos[key][0].append(value_a)
        videos[key][1].append(value_b)

    groups = defaultdict(lambda: [[], []])
    for (category, group, _), (values_a, values_b) in videos.items():
        groups[(category, group)][0].append(np.mean(values_a, axis=0))
        groups[(category, group)][1].append(np.mean(values_b, axis=0))

    by_class = defaultdict(list)
    for (category, group), (values_a, values_b) in groups.items():
        by_class[category].append({
            "group": group,
            "a": np.mean(values_a, axis=0),
            "b": np.mean(values_b, axis=0),
        })
    return {category: sorted(values, key=lambda item: item["group"])
            for category, values in sorted(by_class.items())}


def _metrics(a, b):
    mean_a = float(np.mean(a))
    mean_b = float(np.mean(b))
    return {
        "mse_a": mean_a,
        "mse_b": mean_b,
        "relative_improvement": float(1.0 - mean_b / mean_a),
    }


def summarize_pair(rows, errors_a, errors_b, seed=2026, bootstraps=5000):
    """Summarize paired denoising errors with a source-group cluster bootstrap.

    The point estimate averages window -> video -> source group -> class, then
    macro-averages classes. Bootstrap samples source groups with replacement
    inside each class. One draw is shared by A, B and all noise levels, retaining
    their pairing. The interval measures validation-set sampling uncertainty for
    one training seed; it does not include variation across model training runs.
    """
    errors_a = np.asarray(errors_a, dtype=np.float64)
    errors_b = np.asarray(errors_b, dtype=np.float64)
    expected = (len(rows), len(NOISE_LEVELS))
    if errors_a.shape != expected or errors_b.shape != expected:
        raise ValueError(f"Expected A/B error arrays with shape {expected}.")
    if not (np.isfinite(errors_a).all() and np.isfinite(errors_b).all()):
        raise ValueError("A/B errors must be finite.")

    grouped = _group_errors(rows, errors_a, errors_b)
    classes = list(grouped)
    group_arrays = {
        category: (
            np.stack([item["a"] for item in grouped[category]]),
            np.stack([item["b"] for item in grouped[category]]),
        )
        for category in classes
    }

    point = {}
    for name, columns in SLICES.items():
        per_class = {}
        class_a, class_b = [], []
        for category in classes:
            values_a, values_b = group_arrays[category]
            metrics = _metrics(values_a[:, columns], values_b[:, columns])
            per_class[category] = metrics
            class_a.append(metrics["mse_a"])
            class_b.append(metrics["mse_b"])
        point[name] = _metrics(class_a, class_b)
        point[name]["per_class"] = per_class

    rng = np.random.default_rng(seed)
    bootstrap_values = {
        name: {key: np.empty(bootstraps, dtype=np.float64)
               for key in ("mse_a", "mse_b", "relative_improvement")}
        for name in SLICES
    }
    for repeat in range(bootstraps):
        sampled = {}
        for category in classes:
            values_a, values_b = group_arrays[category]
            indices = rng.integers(0, len(values_a), size=len(values_a))
            sampled[category] = (values_a[indices], values_b[indices])
        for name, columns in SLICES.items():
            class_a, class_b = [], []
            for category in classes:
                values_a, values_b = sampled[category]
                class_a.append(float(np.mean(values_a[:, columns])))
                class_b.append(float(np.mean(values_b[:, columns])))
            metrics = _metrics(class_a, class_b)
            for key, value in metrics.items():
                bootstrap_values[name][key][repeat] = value

    summaries = {}
    for name in SLICES:
        summaries[name] = point[name]
        summaries[name]["ci95"] = {
            key: [float(value) for value in np.quantile(samples, (0.025, 0.975))]
            for key, samples in bootstrap_values[name].items()
        }

    return {
        "primary": summaries["all"],
        "slices": summaries,
        "noise_levels": list(NOISE_LEVELS),
        "class_count": len(classes),
        "group_count": int(sum(len(grouped[category]) for category in classes)),
        "groups_per_class": {category: len(grouped[category]) for category in classes},
        "bootstrap": {
            "seed": seed,
            "replicates": bootstraps,
            "cluster": "paired source groups resampled within class",
        },
        "scope": (
            "Validation-set source-group sampling uncertainty for one training seed; "
            "the interval does not measure variation across training runs."
        ),
    }
