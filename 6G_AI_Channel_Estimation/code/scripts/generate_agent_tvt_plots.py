"""Generate per-agent train/validation/test bar plots for the dashboard."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
ACCENT, GOLD, GREEN, PINK, ORANGE = "#00C9FF", "#F5C451", "#3DDC97", "#FF6B9D", "#FF8A3D"

plt.rcParams.update({
    "figure.facecolor": "#0b1220",
    "axes.facecolor": "#10192b",
    "axes.edgecolor": "#3d4f6f",
    "axes.labelcolor": "#d7e3f4",
    "xtick.color": "#9bb0c9",
    "ytick.color": "#9bb0c9",
    "text.color": "#e8eef7",
    "savefig.facecolor": "#0b1220",
})


def main() -> None:
    report = json.loads((ROOT / "outputs" / "reports" / "train_val_test_report.json").read_text(encoding="utf-8"))
    out = ROOT / "outputs" / "plots" / "agents"
    out.mkdir(parents=True, exist_ok=True)
    arch = report.get("architecture", {})
    n = 0
    for name, payload in report.get("agents", {}).items():
        metrics = payload.get("metrics", {})
        fig, ax = plt.subplots(figsize=(6.2, 3.6))
        importances = {k.replace("importance_", ""): v for k, v in metrics.items() if k.startswith("importance_")}
        if name == "channel":
            ax.bar(["LS", "MMSE", "AI"], [arch.get("test_nmse_ls", 0), arch.get("test_nmse_mmse", 0), arch.get("test_nmse_ai", 0)], color=[PINK, ORANGE, ACCENT])
            ax.set_title("channel — physics NMSE (held-out test)")
            ax.set_ylabel("NMSE")
        elif importances:
            items = sorted(importances.items(), key=lambda kv: -float(kv[1] or 0))
            ax.barh([k for k, _ in items][::-1], [v for _, v in items][::-1], color=ACCENT)
            ax.set_title("explainability — permutation importance")
        else:
            labels, vals, colors = [], [], []
            for lab, key, color in (("train", "train", ACCENT), ("validation", "validation", GOLD), ("test", "test", GREEN)):
                value = None
                for kind in ("accuracy", "r2", "mean_acc", "mean_accuracy", "ho_success", "mean_fidelity", "success_rate"):
                    if metrics.get(f"{key}_{kind}") is not None:
                        value = metrics[f"{key}_{kind}"]
                        break
                if value is None:
                    value = metrics.get(f"binary_{key}_accuracy", metrics.get(f"multiclass_{key}_accuracy", metrics.get(f"ensemble_{key}_r2")))
                if value is not None:
                    labels.append(lab)
                    vals.append(float(value))
                    colors.append(color)
            if not labels and metrics.get("test_mean_nmse_ai") is not None:
                labels, vals, colors = ["test NMSE AI"], [float(metrics["test_mean_nmse_ai"])], [ACCENT]
            if labels:
                ax.bar(labels, vals, color=colors)
                ax.set_title(f"{name} — train / validation / test")
            else:
                ax.text(0.5, 0.5, "no numeric TVT", ha="center", va="center", transform=ax.transAxes)
                ax.set_title(name)
        fig.tight_layout()
        fig.savefig(out / f"{name}_tvt.png", dpi=140, bbox_inches="tight")
        plt.close(fig)
        n += 1
        print(f"wrote {name}_tvt.png")
    print(f"done {n}")


if __name__ == "__main__":
    main()
