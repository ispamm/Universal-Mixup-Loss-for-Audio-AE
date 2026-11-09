#!/usr/bin/env python3
"""
Extract the final hyperparameter-search deliverable from results.json + trial logs.

Usage:
    uv run python training/hparam_report.py

Prints:
  1. A comprehensive overview table of every recorded trial (hyperparameters,
     val_total, val_mag, val_phase (=0, phase loss off), val_mixup, loss_D,
     execution time, status).
  2. Topological-additivity metrics (fast_eval=false fine runs) parsed from
     the "[TopoEval ...]" lines in the trial logs.
  3. The best config dict + checkpoint path.
"""
import json
import re
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
LOG_DIR = REPO / "training" / "checkpoints" / "hparam_search"
RESULTS = LOG_DIR / "results.json"

RE_TOPO = re.compile(
    r"\[TopoEval epoch\s+\d+\]\s+dec_cosine=([-+0-9.eE]+)\s+"
    r"enc_mse=([-+0-9.eE]+)\s+lat_density=([-+0-9.eE]+)\s+"
    r"lat_convexity=([-+0-9.eE]+)"
)


def fmt(v, nd=4):
    if v is None:
        return "—"
    try:
        return f"{float(v):.{nd}f}"
    except (TypeError, ValueError):
        return str(v)


def main() -> None:
    if not RESULTS.exists():
        print("No results.json yet.")
        return
    results = json.loads(RESULTS.read_text())

    def sort_key(r):
        if r.get("status") == "completed":
            return (0, float(r.get("best_val_total", 1e9) or 1e9))
        if r.get("status") == "pruned":
            return (1, float(r.get("last_val_total", 1e9) or 1e9))
        return (2, 1e9)

    ordered = sorted(results, key=sort_key)

    print("\n## Comprehensive overview of all trials\n")
    print("| trial | lr | bs | adv | mixup | K | epochs | val_total | val_mag | val_phase | val_mixup | loss_D | time(min) | status | failure |")
    print("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for r in ordered:
        print(
            f"| {r['name']} | {r['lr']} | {r['batch_size']} | {r['adv_loss_weight']} | "
            f"{r['mixup_weight']} | {r['decoder_num_snake_layers']} | {r.get('epochs_run','—')} | "
            f"{fmt(r.get('best_val_total'))} | {fmt(r.get('best_val_mag'))} | 0.0 | "
            f"{fmt(r.get('best_val_mixup'))} | {fmt(r.get('last_D'))} | "
            f"{fmt((r.get('duration_s') or 0)/60.0, 1)} | {r['status']} | {r.get('failure','—')} |"
        )

    print("\n## Topological additivity (fine runs, fast_eval=false)\n")
    for r in ordered:
        log_path = LOG_DIR / f"{r['name']}.log"
        if not log_path.exists():
            continue
        text = log_path.read_text(errors="replace")
        topo = RE_TOPO.findall(text)
        if topo:
            last = topo[-1]
            print(f"{r['name']}: dec_cosine_sim={fmt(last[0])} enc_mse={fmt(last[1])} "
                  f"lat_density={fmt(last[2])} lat_convexity={fmt(last[3])} (final TopoEval)")

    completed = [r for r in results if r["status"] == "completed"
                 and r.get("best_val_total") is not None]
    if not completed:
        print("\nNo completed trials.")
        return
    best = min(completed, key=lambda r: r["best_val_total"])

    print("\n## Best configuration\n")
    print(f"trial          : {best['name']}")
    print(f"val_total      : {fmt(best.get('best_val_total'))}")
    print(f"val_mag        : {fmt(best.get('best_val_mag'))}")
    print(f"val_mixup      : {fmt(best.get('best_val_mixup'))}")
    print(f"last loss_D    : {fmt(best.get('last_D'))}")
    print(f"best epoch     : {best.get('best_epoch')}")
    print(f"checkpoint     : training/checkpoints/hparam_search/{best['name']}_best.pt")
    print("config dict    :", json.dumps({
        "model_type": "transformer",
        "activation": "leaky_relu",
        "decoder_num_snake_layers": best["decoder_num_snake_layers"],
        "encoder_final_norm": False,
        "decoder_final_norm": False,
        "use_soft_norm": True,
        "lr": best["lr"],
        "batch_size": best["batch_size"],
        "adv_loss_weight": best["adv_loss_weight"],
        "mixup_weight": best["mixup_weight"],
        "soft_norm_kl_weight": 1.0,
        "use_gan": True,
        "use_latent_mixup": True,
        "use_soft_norm": True,
        "use_phase_loss": False,
        "sample_rate": 16000,
    }))


if __name__ == "__main__":
    main()
