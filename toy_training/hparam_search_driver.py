#!/usr/bin/env python3
"""
Autonomous hyperparameter-search driver for ``toy_training/train_toy.py``.

Runs a queue of trials across exactly two GPU slots (cuda:1 and cuda:3 — never
cuda:0 / cuda:2), parses training logs, applies the pruning + failure decision
rules, updates ``PROJECT_STATE.md`` with a live leaderboard, sends Telegram
milestone alerts, and — after the coarse queue is exhausted — automatically
fine-tunes the top-k completed configs.

Resumable: trials whose ``name`` is already recorded in ``results.json``
(status ``completed`` / ``pruned`` / ``failed``) are skipped on restart.

Usage
-----
    uv run python toy_training/hparam_search_driver.py \
        --queue toy_training/hparam_search_queue.json \
        --budget-hours 11 \
        --max-epochs 30 --limit-train-batches 400 --prune-epoch 15 \
        --fine-topk 2 --fine-epochs 60 --fine-limit 1200 \
        --dry-run
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import math
import os
import re
import signal
import subprocess
import sys
import time
from pathlib import Path

# ---------------------------------------------------------------------------
#  Paths / constants
# ---------------------------------------------------------------------------
REPO = Path(__file__).resolve().parents[1]
LOG_DIR = REPO / "toy_training" / "checkpoints" / "hparam_search"
RESULTS_PATH = LOG_DIR / "results.json"
STATE_PATH = REPO / "PROJECT_STATE.md"
TELEGRAM = str(Path.home() / ".agents" / "skills" / "agent-orchestrator" / "telegram_notify.py")
CACHE_MANIFEST = Path("/data/preprocessed_cache/sr16000/manifest.json")

DATA_DIRS = [
    "/data/wav48_silence_trimmed",
    "/data/fsd50k/FSD50K.dev_audio",
    "/data/fma_small",
]

GPU_SLOTS = ["cuda:1", "cuda:3"]

# Fixed flags shared by every trial (per the search brief).
FIXED_FLAGS = {
    "model_type": "transformer",
    "activation": "leaky_relu",
    "sample_rate": 16000,
    "use_gan": True,
    "use_latent_mixup": True,
    "use_soft_norm": True,
    "use_phase_loss": False,
    "use_morph_gan": False,
    "soft_norm_kl_weight": 1.0,
    "patience": 15,
    "min_delta": 1e-5,
    "num_workers": 4,
    "seed": 42,
    "chunk_seconds": 2.0,
    "preprocess_cache_dir": "/data/preprocessed_cache",
}

# --- Log parsing regexes ----------------------------------------------------
RE_EPOCH = re.compile(r"\[Epoch\s+(\d+)/(\d+)\]")
RE_VAL = re.compile(
    r"\|\s*val:\s*total=([-+0-9.eE]+)\s*mag=([-+0-9.eE]+)(?:\s*mixup=([-+0-9.eE]+))?"
)
RE_D = re.compile(r"\bD=([-+0-9.eE]+)")
RE_BEST = re.compile(r"\[Best\] Best validation loss: ([-+0-9.eE]+)")


# ---------------------------------------------------------------------------
#  Helpers
# ---------------------------------------------------------------------------
def now_iso() -> str:
    return dt.datetime.now().isoformat(timespec="seconds")


def log(msg: str) -> None:
    print(f"[driver {now_iso()}] {msg}", flush=True)


def notify(msg: str) -> None:
    try:
        subprocess.run(
            ["python", TELEGRAM, msg],
            check=False, timeout=60,
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
    except Exception as exc:  # noqa: BLE001
        log(f"telegram notify failed: {exc}")


def is_float_finite(x) -> bool:
    return x is not None and math.isfinite(float(x))


# ---------------------------------------------------------------------------
#  Command construction
# ---------------------------------------------------------------------------
def build_cmd(trial: dict) -> list[str]:
    t = {**FIXED_FLAGS, **trial}
    cmd = [
        "uv", "run", "python", "toy_training/train_toy.py",
        "--data_dirs", *DATA_DIRS,
        "--run_name", f"hparam_search/{t['name']}",
        "--model_type", t["model_type"],
        "--activation", t["activation"],
        "--sample_rate", str(t["sample_rate"]),
        "--device", t["device"],
        "--lr", str(t["lr"]),
        "--batch_size", str(t["batch_size"]),
        "--epochs", str(t["epochs"]),
        "--adv_loss_weight", str(t["adv_loss_weight"]),
        "--mixup_weight", str(t["mixup_weight"]),
        "--decoder_num_snake_layers", str(t["decoder_num_snake_layers"]),
        "--soft_norm_kl_weight", str(t["soft_norm_kl_weight"]),
        "--patience", str(t["patience"]),
        "--min_delta", str(t["min_delta"]),
        "--num_workers", str(t["num_workers"]),
        "--seed", str(t["seed"]),
        "--chunk_seconds", str(t["chunk_seconds"]),
        "--preprocess_cache_dir", t["preprocess_cache_dir"],
    ]
    if t.get("limit_train_batches", 0) > 0:
        cmd += ["--limit_train_batches", str(t["limit_train_batches"])]
    if t["use_gan"]:
        cmd.append("--use_gan")
    if t["use_latent_mixup"]:
        cmd.append("--use_latent_mixup")
    if t["use_soft_norm"]:
        cmd.append("--use_soft_norm")
    if t["use_phase_loss"]:
        cmd.append("--use_phase_loss")
    if t.get("fast_eval", True):
        cmd.append("--fast_eval")
    return cmd


# ---------------------------------------------------------------------------
#  Log parsing / failure detection
# ---------------------------------------------------------------------------
def parse_log(path: Path) -> tuple[list[dict], float | None]:
    records: list[dict] = []
    best: float | None = None
    if not path.exists():
        return records, best
    text = path.read_text(errors="replace")
    for ln in text.splitlines():
        m = RE_EPOCH.search(ln)
        if not m:
            continue
        epoch = int(m.group(1))
        vm = RE_VAL.search(ln)
        if vm is None:
            continue  # truncated/incomplete line; complete on next poll
        dm = RE_D.search(ln)
        val_total = float(vm.group(1))
        val_mag = float(vm.group(2))
        val_mixup = float(vm.group(3)) if vm.group(3) else 0.0
        D = float(dm.group(1)) if dm else None
        records.append({
            "epoch": epoch, "val_total": val_total,
            "val_mag": val_mag, "val_mixup": val_mixup, "D": D,
        })
    bm = RE_BEST.search(text)
    if bm:
        best = float(bm.group(1))
    return records, best


def detect_failure(path: Path, records: list[dict]) -> str | None:
    text = path.read_text(errors="replace") if path.exists() else ""
    low = text.lower()
    if "cuda out of memory" in low or "outofmemoryerror" in low:
        return "CUDA OOM"
    if "cublas" in low and ("error" in low or "not supported" in low):
        return "CUBLAS error"

    for r in records:
        if not is_float_finite(r["val_total"]):
            return "NaN/Inf loss"

    for r in records:
        d = r.get("D")
        if d is None:
            continue
        if d <= 1e-9 or d > 1.5:
            return f"Discriminator out-of-range (D={d:.4f})"

    ep5 = next((r for r in records if r["epoch"] == 5), None)
    if ep5 is not None:
        ref = ep5["val_total"]
        if is_float_finite(ref) and float(ref) > 0:
            last = records[-1]["val_total"]
            if is_float_finite(last) and float(last) > 2.5 * float(ref):
                return f"val_total diverged (ep{records[-1]['epoch']}={last:.4f} vs ep5={ref:.4f})"

    return None


# ---------------------------------------------------------------------------
#  Leaderboard / project state
# ---------------------------------------------------------------------------
def _fmt(v, nd=4) -> str:
    if v is None:
        return "—"
    try:
        return f"{float(v):.{nd}f}"
    except (TypeError, ValueError):
        return str(v)


def update_project_state(results: list[dict]) -> None:
    START = "<!-- HPARAM_SEARCH_START -->"
    END = "<!-- HPARAM_SEARCH_END -->"
    state = STATE_PATH.read_text(errors="replace") if STATE_PATH.exists() else ""

    def sort_key(r):
        if r.get("status") == "completed":
            return (0, float(r.get("best_val_total", 1e9) or 1e9))
        if r.get("status") == "pruned":
            return (1, float(r.get("last_val_total", 1e9) or 1e9))
        return (2, 1e9)

    ordered = sorted(results, key=sort_key)
    lines = [
        START, "",
        "## Hyperparameter Search — Live Leaderboard", "",
        f"_Auto-generated by hparam_search_driver.py · last update {now_iso()} · "
        f"{len(ordered)} trials recorded_", "",
        "| # | trial | lr | bs | λ_adv | λ_mixup | K | epochs | val_total | val_mag | val_mixup | loss_D | time | status |",
        "|---|-------|----|----|-------|---------|---|--------|-----------|---------|-----------|--------|------|--------|",
    ]
    for i, r in enumerate(ordered, 1):
        lines.append(
            f"| {i} | {r['name']} | {r['lr']} | {r['batch_size']} | "
            f"{r['adv_loss_weight']} | {r['mixup_weight']} | {r['decoder_num_snake_layers']} | "
            f"{r.get('epochs_run', '—')} | {_fmt(r.get('best_val_total'))} | "
            f"{_fmt(r.get('best_val_mag'))} | {_fmt(r.get('best_val_mixup'))} | "
            f"{_fmt(r.get('last_D'))} | {_fmt((r.get('duration_s') or 0) / 60.0, 1)}m | "
            f"{r.get('status')} |"
        )
    lines += ["", END, ""]
    block = "\n".join(lines)

    if START in state and END in state:
        new_state = state[: state.index(START)] + block + state[state.index(END) + len(END):]
    else:
        new_state = state.rstrip("\n") + "\n\n" + block + "\n"
    STATE_PATH.write_text(new_state, encoding="utf-8")


def load_results() -> list[dict]:
    if RESULTS_PATH.exists():
        try:
            return json.loads(RESULTS_PATH.read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001
            return []
    return []


def save_results(results: list[dict]) -> None:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    tmp = RESULTS_PATH.with_suffix(".tmp")
    tmp.write_text(json.dumps(results, indent=2), encoding="utf-8")
    tmp.replace(RESULTS_PATH)


# ---------------------------------------------------------------------------
#  Worker
# ---------------------------------------------------------------------------
class Worker:
    def __init__(self, device: str):
        self.device = device
        self.trial: dict | None = None
        self.proc: subprocess.Popen | None = None
        self.log_path: Path | None = None
        self.start: float = 0.0
        self.epochs_requested: int = 0
        self.fast_eval: bool = True
        self.status: str | None = None
        self.failure: str | None = None
        self.records: list[dict] = []
        self.best: float | None = None
        self._fh = None

    @property
    def busy(self) -> bool:
        return self.proc is not None and self.proc.poll() is None

    def launch(self, trial: dict) -> None:
        self.trial = trial
        self.epochs_requested = trial["epochs"]
        self.fast_eval = trial.get("fast_eval", True)
        self.records = []
        self.best = None
        self.status = "running"
        self.failure = None
        self.log_path = LOG_DIR / f"{trial['name']}.log"
        trial = {**trial, "device": self.device}
        cmd = build_cmd(trial)
        log(f"[{self.device}] launch {trial['name']} epochs={self.epochs_requested} "
            f"limit={trial.get('limit_train_batches', 0)} fast_eval={self.fast_eval}")
        self._fh = open(self.log_path, "w")
        self._fh.write(f"# {trial['name']} on {self.device}\n")
        self._fh.write(f"# cmd: {' '.join(cmd)}\n\n")
        self._fh.flush()
        self.proc = subprocess.Popen(
            cmd,
            cwd=str(REPO),
            stdout=self._fh,
            stderr=subprocess.STDOUT,
            env={**os.environ, "WANDB_MODE": "disabled"},
            preexec_fn=os.setsid,
        )
        self.start = time.time()

    def terminate(self) -> None:
        if self.proc is not None and self.proc.poll() is None:
            try:
                os.killpg(os.getpgid(self.proc.pid), signal.SIGTERM)
            except Exception:  # noqa: BLE001
                self.proc.terminate()
            try:
                self.proc.wait(timeout=30)
            except subprocess.TimeoutExpired:
                try:
                    os.killpg(os.getpgid(self.proc.pid), signal.SIGKILL)
                except Exception:  # noqa: BLE001
                    self.proc.kill()

    def poll_log(self) -> None:
        if self.log_path is not None:
            self.records, self.best = parse_log(self.log_path)

    def finalize(self, results: list[dict], status: str) -> None:
        if self.trial is None:
            return
        duration = time.time() - self.start
        rec = self.records
        last_D = rec[-1]["D"] if rec and rec[-1].get("D") is not None else None
        last_val = rec[-1]["val_total"] if rec else None

        best_idx = None
        best_val_total = None
        if rec:
            valid = [i for i, r in enumerate(rec) if is_float_finite(r["val_total"])]
            if valid:
                best_idx = min(valid, key=lambda i: rec[i]["val_total"])
                best_val_total = rec[best_idx]["val_total"]

        entry = {
            "name": self.trial["name"],
            "device": self.device,
            "lr": self.trial["lr"],
            "batch_size": self.trial["batch_size"],
            "adv_loss_weight": self.trial["adv_loss_weight"],
            "mixup_weight": self.trial["mixup_weight"],
            "decoder_num_snake_layers": self.trial["decoder_num_snake_layers"],
            "epochs_requested": self.epochs_requested,
            "limit_train_batches": self.trial.get("limit_train_batches", 0),
            "fast_eval": self.fast_eval,
            "epochs_run": len(rec),
            "status": status,
            "failure": self.failure,
            "best_val_total": best_val_total,
            "best_val_mag": rec[best_idx]["val_mag"] if best_idx is not None else None,
            "best_val_mixup": rec[best_idx]["val_mixup"] if best_idx is not None else None,
            "best_epoch": rec[best_idx]["epoch"] if best_idx is not None else None,
            "last_val_total": last_val,
            "last_D": last_D,
            "duration_s": round(duration, 1),
            "finished_at": now_iso(),
        }
        results[:] = [r for r in results if r["name"] != self.trial["name"]]
        results.append(entry)

        log(f"[{self.device}] {self.trial['name']} -> {status}"
            + (f" ({self.failure})" if self.failure else "")
            + f" | best_val_total={_fmt(best_val_total)} last_D={_fmt(last_D)} "
            + f"epochs={len(rec)} time={duration/60:.1f}m")

        if self._fh is not None:
            try:
                self._fh.close()
            except Exception:  # noqa: BLE001
                pass
            self._fh = None
        self.proc = None
        self.trial = None
        self.log_path = None
        self.status = None
        self.failure = None
        self.records = []
        self.best = None


# ---------------------------------------------------------------------------
#  Decision rules
# ---------------------------------------------------------------------------
def pruning_baseline(results: list[dict]) -> list[float] | None:
    vals = [r["best_val_total"] for r in results
            if r.get("status") == "completed" and is_float_finite(r.get("best_val_total"))]
    return vals if len(vals) >= 3 else None


def should_prune(worker: Worker, baseline: list[float], prune_epoch: int) -> bool:
    if not worker.records:
        return False
    last = worker.records[-1]
    if last["epoch"] < prune_epoch:
        return False
    if not is_float_finite(last["val_total"]):
        return False
    sorted_vals = sorted(baseline)
    idx = int(0.7 * (len(sorted_vals) - 1))
    return last["val_total"] > sorted_vals[idx]


def make_fine_trials(results: list[dict], args) -> list[dict]:
    completed = [r for r in results
                 if r.get("status") == "completed" and is_float_finite(r.get("best_val_total"))]
    if not completed:
        return []
    completed.sort(key=lambda r: r["best_val_total"])
    top = completed[: args.fine_topk]
    trials = []
    for r in top:
        trials.append({
            "name": f"{r['name']}-fine",
            "lr": r["lr"],
            "batch_size": r["batch_size"],
            "adv_loss_weight": r["adv_loss_weight"],
            "mixup_weight": r["mixup_weight"],
            "decoder_num_snake_layers": r["decoder_num_snake_layers"],
            "epochs": args.fine_epochs,
            "limit_train_batches": args.fine_limit,
            "fast_eval": args.fine_fast_eval,
            "prune": False,
        })
    return trials


# ---------------------------------------------------------------------------
#  Main loop
# ---------------------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--queue", required=True)
    ap.add_argument("--budget-hours", type=float, default=11.0)
    ap.add_argument("--max-epochs", type=int, default=30)
    ap.add_argument("--limit-train-batches", type=int, default=0)
    ap.add_argument("--prune-epoch", type=int, default=15)
    ap.add_argument("--fast-eval", dest="fast_eval", action="store_true", default=True)
    ap.add_argument("--no-fast-eval", dest="fast_eval", action="store_false")
    ap.add_argument("--fine-topk", type=int, default=2)
    ap.add_argument("--fine-epochs", type=int, default=60)
    ap.add_argument("--fine-limit", type=int, default=1200)
    ap.add_argument("--fine-fast-eval", dest="fine_fast_eval", action="store_true", default=False)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--notify-every", type=int, default=4)
    args = ap.parse_args()

    queue = json.loads(Path(args.queue).read_text(encoding="utf-8"))
    if isinstance(queue, dict):
        queue = queue.get("trials", [])

    # Normalise each trial with driver-level defaults.
    for t in queue:
        t.setdefault("epochs", args.max_epochs)
        t.setdefault("fast_eval", args.fast_eval)
        t.setdefault("limit_train_batches", args.limit_train_batches)
        t.setdefault("prune", True)

    results = load_results()
    done_names = {r["name"] for r in results
                  if r.get("status") in ("completed", "pruned", "failed")}
    pending = [t for t in queue if t["name"] not in done_names]

    log(f"queue={len(queue)} pending={len(pending)} done={len(done_names)} "
        f"budget={args.budget_hours}h max_epochs={args.max_epochs} "
        f"limit={args.limit_train_batches} prune_epoch={args.prune_epoch}")

    if args.dry_run:
        for t in pending:
            print(" ".join(build_cmd({**t, "device": GPU_SLOTS[0]})))
        return 0

    budget_s = args.budget_hours * 3600.0
    consumed_s = sum(r.get("duration_s", 0) for r in results
                     if r.get("status") in ("completed", "pruned", "failed"))
    workers = [Worker(g) for g in GPU_SLOTS]
    finished_since_notify = 0
    fine_queued = args.fine_topk <= 0  # skip fine phase if disabled

    notify("Overnight hyperparameter search driver started. "
           f"{len(pending)} pending trials, budget {args.budget_hours}h, "
           "workers on cuda 1 and cuda 3.")

    while True:
        if not CACHE_MANIFEST.exists():
            log("waiting for preprocessed cache manifest ...")
            time.sleep(30)
            continue

        for w in workers:
            if not w.busy and w.trial is None and pending and consumed_s < budget_s:
                w.launch(pending.pop(0))

        active = [w for w in workers if w.busy]
        if not active and not pending:
            if not fine_queued:
                fine_queued = True
                fine_trials = make_fine_trials(results, args)
                if fine_trials:
                    pending = fine_trials
                    log(f"coarse done -> queuing {len(pending)} fine trials: "
                        f"{[t['name'] for t in pending]}")
                    notify("Coarse search finished. Now fine-tuning the top "
                           f"{len(pending)} configurations at longer horizon.")
                    continue
            log("queue exhausted")
            break
        if not active and pending and consumed_s >= budget_s:
            log(f"budget exhausted ({consumed_s/3600:.2f}h); {len(pending)} trials skipped")
            break
        if not active and not pending and consumed_s >= budget_s:
            break

        for w in workers:
            if w.proc is None or w.trial is None:
                continue  # idle worker, already finalized

            # ---- normal completion (process exited on its own) ----
            rc = w.proc.poll()
            if rc is not None:
                status = "completed" if rc == 0 else "failed"
                if status == "failed" and w.failure is None:
                    w.failure = f"exit code {rc}"
                w.finalize(results, status)
                finished_since_notify += 1
                consumed_s += results[-1]["duration_s"]
                save_results(results)
                update_project_state(results)
                continue

            # ---- still running: poll log for failure / pruning ----
            w.poll_log()
            fail = detect_failure(w.log_path, w.records)
            if fail and w.failure is None:
                w.failure = fail
                log(f"[{w.device}] {w.trial['name']} FAILURE: {fail}")
                w.terminate()
                w.finalize(results, "failed")
                finished_since_notify += 1
                save_results(results)
                update_project_state(results)
                continue

            if w.trial.get("prune", True):
                base = pruning_baseline(results)
                if base is not None and should_prune(w, base, args.prune_epoch):
                    log(f"[{w.device}] {w.trial['name']} PRUNED at epoch "
                        f"{w.records[-1]['epoch']} (val_total={w.records[-1]['val_total']:.4f})")
                    w.terminate()
                    w.finalize(results, "pruned")
                    finished_since_notify += 1
                    save_results(results)
                    update_project_state(results)
                    continue

        if finished_since_notify >= args.notify_every:
            done = [r for r in results if r["status"] in ("completed", "pruned", "failed")]
            best = min((r for r in results if r["status"] == "completed"
                        and is_float_finite(r.get("best_val_total"))),
                       key=lambda r: r["best_val_total"], default=None)
            msg = f"Hyperparameter search progress: {len(done)} trials done. "
            if best is not None:
                msg += (f"Best so far {best['name']} val total {best['best_val_total']:.4f} "
                        f"lr {best['lr']} adv {best['adv_loss_weight']} "
                        f"mixup {best['mixup_weight']} K {best['decoder_num_snake_layers']}.")
            notify("🚀 " + msg)
            finished_since_notify = 0

        time.sleep(15)

    update_project_state(results)
    completed = [r for r in results if r["status"] == "completed"
                 and is_float_finite(r.get("best_val_total"))]
    best = min(completed, key=lambda r: r["best_val_total"]) if completed else None
    if best is not None:
        notify("🏁 Hyperparameter search finished. "
               f"Best config {best['name']} val total {best['best_val_total']:.4f} "
               f"lr {best['lr']} bs {best['batch_size']} adv {best['adv_loss_weight']} "
               f"mixup {best['mixup_weight']} K {best['decoder_num_snake_layers']}. "
               "Full report in PROJECT STATE file.")
    else:
        notify("🏁 Hyperparameter search finished with no healthy completed trials. "
               "See PROJECT STATE file for details.")
    log("driver finished")
    return 0


if __name__ == "__main__":
    sys.exit(main())
