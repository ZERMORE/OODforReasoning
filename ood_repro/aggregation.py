"""CPU-only confidence aggregation and sequence-order controls (E1/E2)."""
import math

import numpy as np


def window_means(values, width):
    x = np.asarray(values, dtype=float)
    if x.ndim != 1 or not len(x) or not np.isfinite(x).all() or width < 1:
        raise ValueError("Need a finite nonempty vector and positive window")
    width = min(width, len(x))
    sums = np.r_[0., x.cumsum()]
    return (sums[width:] - sums[:-width]) / width


def bottom_mean(values):
    count = max(1, math.ceil(.1 * len(values)))
    return float(np.partition(values, count - 1)[:count].mean())


def aggregate(values, widths=(32, 64, 128), steps=None):
    x = np.asarray(values, dtype=float)
    window_means(x, 1)  # Validate also when widths is empty.
    result = {"mean": float(x.mean()), "min": float(x.min()), "token_bottom10": bottom_mean(x)}
    position = np.linspace(0, 1, len(x))
    for name, sign in (("head_weighted", -1), ("tail_weighted", 1)):
        weights = np.exp(sign * 2 * position)
        result[name] = float(weights @ x / weights.sum())
    for width in widths:
        windows = window_means(x, width)
        result[f"lowest_w{width}"] = float(windows.min())
        result[f"bottom10_w{width}"] = bottom_mean(windows)
    if steps is not None:
        # Steps must partition all SCORED positions exactly, not overlap or omit text.
        coverage = [i for start, end in steps for i in range(start, end)]
        if coverage != list(range(len(x))) or any(start >= end for start, end in steps):
            raise ValueError("Step ranges must partition scored positions in order")
        result["step_equal_mean"] = float(np.mean([x[start:end].mean() for start, end in steps]))
    return result


def permutation_indices(length, rng, mode="shuffle", block=16):
    if length < 1 or block < 1:
        raise ValueError("Invalid permutation dimensions")
    ids = np.arange(length)
    if mode == "shuffle":
        return rng.permutation(ids)
    if mode == "phase":
        return np.concatenate([rng.permutation(group) for group in np.array_split(ids, 3)])
    if mode == "block":
        groups = [ids[start:start + block] for start in range(0, length, block)]
        return np.concatenate([groups[i] for i in rng.permutation(len(groups))])
    if mode == "reverse":
        return ids[::-1]
    raise ValueError(f"Unknown permutation control: {mode}")
