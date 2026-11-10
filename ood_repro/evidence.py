"""Evidence Drop on confidence trajectories and risk/coverage summaries."""
import numpy as np


def evidence_drop(values, span=5, worst=5):
    """Return positive risk and zero-based right endpoints of selected drops."""
    x = np.asarray(values, dtype=float)
    if x.ndim != 1 or not len(x) or not np.isfinite(x).all() or span < 1 or worst < 1:
        raise ValueError("Need a finite nonempty trajectory and positive span and worst count")
    smooth = x.copy()
    alpha = 2 / (span + 1)
    for i in range(1, len(x)):
        smooth[i] = alpha * x[i] + (1 - alpha) * smooth[i - 1]
    changes = np.diff(smooth)
    positions = np.flatnonzero(changes < 0)
    selected = positions[np.argsort(changes[positions], kind="stable")[:worst]]
    return (-float(changes[selected].mean()) if len(selected) else 0., (selected + 1).tolist())


def fixed_scores(arrays):
    from .aggregation import aggregate
    pooled = [aggregate(x[:, 0], widths=(64,)) for x in arrays]
    result = {name: np.array([row[name] for row in pooled]) for name in ("mean", "token_bottom10", "lowest_w64")}
    drops = [evidence_drop(x[:, 0]) for x in arrays]
    result["evidence_drop"] = -np.array([risk for risk, _ in drops])
    return result, [positions for _, positions in drops]


def risk_curve(errors, confidence):
    """Keep tied scores together so the curve is independent of row order."""
    order = np.argsort(-confidence, kind="stable")
    accepted = np.r_[np.flatnonzero(np.diff(confidence[order])) + 1, len(order)]
    return accepted / len(order), np.cumsum(errors[order])[accepted - 1] / accepted


def detection_summary(errors, confidence, calibration_errors, calibration):
    from sklearn.metrics import average_precision_score
    coverage, risk = risk_curve(errors, confidence)
    result = {"ap": float(average_precision_score(errors, -confidence)),
              "aurc": float(np.diff(np.r_[0., coverage]) @ risk), "operating_points": {}}
    for alpha in (.05, .1, .5):
        # Accept strictly above the threshold. Ties at the cutoff are rejected.
        threshold = float(np.quantile(calibration[calibration_errors == 1], 1-alpha, method="higher"))
        accepted = confidence > threshold
        result["operating_points"][str(alpha)] = {
            "confidence_threshold": threshold,
            "calibration_wrong_acceptance": float((calibration[calibration_errors == 1] > threshold).mean()),
            "test_wrong_acceptance": float(accepted[errors == 1].mean()),
            "test_fpr": float((~accepted[errors == 0]).mean()),
            "coverage": float(accepted.mean()),
            "selective_accuracy": float(1-errors[accepted].mean()) if accepted.any() else None,
        }
    return result


def render_distributions(models, errors, output):
    import matplotlib.pyplot as plt
    from .visualize import save_figure
    names = [name for name in models if "_test_" not in name and not name.endswith("/length_only")]
    signals = list(dict.fromkeys(name.split("/")[0] for name in names))
    methods = list(dict.fromkeys(name.split("/")[1] for name in names))
    for curve in ("ecdf", "risk_coverage"):
        fig, axes = plt.subplots(len(signals), len(methods), squeeze=False,
                                 figsize=(3.5*len(methods), 2.8*len(signals)))
        for i, signal in enumerate(signals):
            for j, method in enumerate(methods):
                ax = axes[i, j]
                replicas = models[f"{signal}/{method}"]
                # Display one named replica rather than averaging logits across seeds.
                seed, confidence = next(iter(replicas.items()))
                if curve == "ecdf":
                    for label, mask in (("Correct", errors == 0), ("Wrong", errors == 1)):
                        values = np.sort(-confidence[mask])
                        ax.step(values, np.arange(1, len(values)+1)/len(values), where="post", label=label)
                    ax.set(xlabel="Risk", ylabel="Cumulative fraction")
                    ax.legend(fontsize=8)
                else:
                    coverage, risk = risk_curve(errors, confidence)
                    ax.step(coverage, risk, where="pre")
                    ax.set(xlabel="Coverage", ylabel="Error rate among accepted", xlim=(0, 1), ylim=(0, 1))
                ax.set_title(f"{signal} / {method}" + (f" (seed {seed})" if seed != "fixed" else ""), fontsize=9)
        fig.tight_layout()
        save_figure(fig, output, curve)
