#!/usr/bin/env python3
"""Collect loss/checkpoint/evaluation locations into one comparison table."""

from __future__ import annotations

import csv
import json
import pathlib
import re

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parents[2]
ROBOTWIN = ROOT.parent / "RoboTwin"


def latest_result(name: str):
    base = ROBOTWIN / "eval_result/place_empty_cup/pi0/demo_aloha" / name
    files = sorted(base.glob("*/_result.txt"), key=lambda p: p.stat().st_mtime)
    if not files:
        return "", ""
    text = files[-1].read_text(encoding="utf-8", errors="replace")
    numbers = re.findall(r"(?m)^([01](?:\.\d+)?)$", text)
    return str(files[-1].parent), (numbers[-1] if numbers else "")


rows = []
for config_path in sorted((HERE / "configs").glob("*.json")):
    cfg = json.loads(config_path.read_text(encoding="utf-8"))
    name = cfg["experiment_id"]
    artifact = ROOT / cfg["output_dir"]
    history = artifact / "epoch_history.csv"
    best_val = ""
    final_train = ""
    if history.is_file():
        with history.open(newline="", encoding="utf-8") as handle:
            records = list(csv.DictReader(handle))
        if records:
            final_train = records[-1]["train_loss"]
            best_val = min(float(row["val_loss"]) for row in records)
    result_dir, success = latest_result(name)
    rows.append({
        "experiment": name,
        "change": cfg["description"],
        "final_train_loss": final_train,
        "best_val_loss": best_val,
        "success_rate": success,
        "checkpoint": str(artifact / "checkpoints/best.pkl"),
        "eval_result_dir": result_dir,
    })

with (HERE / "summary.csv").open("w", newline="", encoding="utf-8") as handle:
    writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
    writer.writeheader()
    writer.writerows(rows)

headers = list(rows[0].keys())
lines = ["# ACT ablation summary", "", "| " + " | ".join(headers) + " |",
         "| " + " | ".join(["---"] * len(headers)) + " |"]
for row in rows:
    lines.append("| " + " | ".join(str(row[key]).replace("|", "\\|") for key in headers) + " |")
(HERE / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
print(HERE / "summary.md")
