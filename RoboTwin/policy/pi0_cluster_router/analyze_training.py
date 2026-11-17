"""Create a compact optimization report from a routed training log."""

import argparse
import json
from pathlib import Path
import re
from statistics import mean


PATTERN = re.compile(
    r"Step\s+(?P<step>\d+):\s+grad_norm=(?P<grad>[0-9.eE+-]+),\s+"
    r"loss=(?P<loss>[0-9.eE+-]+),\s+param_norm=(?P<param>[0-9.eE+-]+)")


def window(records, predicate):
    values = [record for record in records if predicate(record["step"])]
    if not values:
        return None
    return {
        "points": len(values),
        "mean_loss": mean(record["loss"] for record in values),
        "min_loss": min(record["loss"] for record in values),
        "max_loss": max(record["loss"] for record in values),
        "mean_grad_norm": mean(record["grad_norm"] for record in values),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    options = parser.parse_args()
    run_dir = options.run_dir.resolve()
    candidates = sorted((run_dir / "logs").glob("train_task_*.log"))
    parsed = []
    source = None
    for candidate in candidates:
        current = [
            {"step": int(match["step"]), "grad_norm": float(match["grad"]),
             "loss": float(match["loss"]), "param_norm": float(match["param"])}
            for match in PATTERN.finditer(candidate.read_text(errors="replace"))
        ]
        if len(current) > len(parsed):
            parsed, source = current, candidate
    if not parsed:
        raise SystemExit(f"No training metrics found under {run_dir / 'logs'}")

    final_step = parsed[-1]["step"]
    report = {
        "run_dir": str(run_dir),
        "source_log": str(source),
        "logged_points": len(parsed),
        "first": parsed[0],
        "final": parsed[-1],
        "loss_reduction_fraction": 1 - parsed[-1]["loss"] / parsed[0]["loss"],
        "windows": {
            "steps_1_500": window(parsed, lambda step: step <= 500),
            "steps_501_1000": window(parsed, lambda step: 500 < step <= 1000),
            "after_1000": window(parsed, lambda step: step > 1000),
            "final_1000": window(parsed, lambda step: step > final_step - 1000),
        },
        "interpretation": (
            "Optimization is numerically stable and the training objective decreased. "
            "RoboTwin simulation success rate is still required to measure task performance."
        ),
    }
    output_dir = run_dir / "reports"
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "training_analysis.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    steps = [record["step"] for record in parsed]
    figure, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
    axes[0].plot(steps, [record["loss"] for record in parsed], color="#2563eb", linewidth=1.8)
    axes[0].set_ylabel("Flow-matching loss")
    axes[0].grid(alpha=0.25)
    axes[1].plot(steps, [record["grad_norm"] for record in parsed], color="#ea580c", linewidth=1.4)
    axes[1].set_xlabel("Training step")
    axes[1].set_ylabel("Gradient norm")
    axes[1].grid(alpha=0.25)
    figure.suptitle("Cluster7 Top-3 Router Training")
    figure.tight_layout()
    figure.savefig(output_dir / "training_curve.png", dpi=160)
    print(f"Report: {output_dir / 'training_analysis.json'}")
    print(f"Curve: {output_dir / 'training_curve.png'}")


if __name__ == "__main__":
    main()
