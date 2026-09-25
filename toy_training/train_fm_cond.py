#!/usr/bin/env python3
"""
Conditional flow matching for latent stem extraction (Exp 4, Task A).

Task.  Given a mixture latent `z_mix` and the interfering stem latent `z_other`,
generate the target voice latent `z_voice`:

    v_theta(z_t, t, z_mix, z_other)  approximating  z_noise - z_voice

with the rectified-flow recipe of stable_audio_3 / train_fm_pilot.py (linear
interpolant, velocity target, minibatch OT coupling).  The target is the AE's own
`encode(voice)`, so the task is well posed in BOTH latent spaces and no additivity
assumption is baked in.

Why this metric ladder is logged during training.  Final separation quality is a
*low-dynamic-range* metric on the additive latent: the algebraic oracle already
sits only ~0.9 dB below the univ AE's own reconstruction ceiling, so an FM that
merely reproduces the latent arithmetic cannot beat it by much.  The evidence has
to come from *dynamics* -- how quickly the FM reaches the oracle, and how few
sampling steps it needs (a straight learned field needs very few; a curved one
needs many).  Therefore every `--eval_every` epochs we log, on fixed val clips:

    eval/sisdr_ceiling      decode(z_voice)              AE upper bound
    eval/sisdr_oracle       decode(z_mix - z_other)      algebraic oracle
    eval/sisdr_noop         decode(z_mix)                trivial floor
    eval/sisdr_fm           decode(sample(FM, steps))    the result
    eval/sisdr_steps_{1,2,4,8,16,32}                     step-count sweep

plus MS-STFT for the same set.  All four baselines are computed once and are
identical for both runs, so the curves are directly comparable.

USAGE
    python toy_training/train_fm_cond.py \
        --mixture_dir toy_evaluation/exp4_mixtures/univ \
        --ae_checkpoint toy_training/checkpoints/transformer_univ_warm_v2_best.pt \
        --run_name fm_cond_univ --device cuda:0
"""
from __future__ import annotations
import argparse, functools, json, sys, time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from toy_evaluation.model_utils import ToyModelWrapper
from toy_training.losses import SAMELoss
from toy_training.train_fm_pilot import ot_coupling
from toy_evaluation.eval_recon_sisdr import si_sdr_db

print = functools.partial(print, flush=True)


# ---------------------------------------------------------------------------
#  Conditional DiT (same body as train_fm_pilot.LatentDiT; the only change is
#  that the input projection accepts [z_t, z_mix, z_other] concatenated on the
#  channel axis, so the parameter count is identical for both latent spaces)
# ---------------------------------------------------------------------------
class DiTBlock(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, elementwise_affine=False)
        self.attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.norm2 = nn.LayerNorm(dim, elementwise_affine=False)
        mlp_dim = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(nn.Linear(dim, mlp_dim), nn.GELU(), nn.Linear(mlp_dim, dim))
        self.adaLN = nn.Sequential(nn.SiLU(), nn.Linear(dim, 6 * dim))

    def forward(self, x, c):
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = \
            self.adaLN(c).chunk(6, dim=-1)
        h = self.norm1(x) * (1 + scale_msa.unsqueeze(1)) + shift_msa.unsqueeze(1)
        x = x + gate_msa.unsqueeze(1) * self.attn(h, h, h, need_weights=False)[0]
        h = self.norm2(x) * (1 + scale_mlp.unsqueeze(1)) + shift_mlp.unsqueeze(1)
        x = x + gate_mlp.unsqueeze(1) * self.mlp(h)
        return x


class CondLatentDiT(nn.Module):
    """DiT over (B, D, T_lat) latents with two conditioning latents."""

    def __init__(self, channels=64, embed_dim=384, depth=6, num_heads=8,
                 mlp_ratio=4.0, t_dim=256, n_cond=2):
        super().__init__()
        self.channels = channels
        self.n_cond = n_cond
        self.in_proj = nn.Linear(channels * (1 + n_cond), embed_dim)
        self.time_embed = nn.Sequential(
            nn.Linear(1, t_dim), nn.SiLU(), nn.Linear(t_dim, t_dim),
            nn.SiLU(), nn.Linear(t_dim, embed_dim))
        self.blocks = nn.ModuleList([
            DiTBlock(embed_dim, num_heads, mlp_ratio) for _ in range(depth)])
        self.final_norm = nn.LayerNorm(embed_dim, elementwise_affine=False)
        self.final_adaLN = nn.Sequential(nn.SiLU(), nn.Linear(embed_dim, 2 * embed_dim))
        self.out_proj = nn.Linear(embed_dim, channels)

    def forward(self, z, t, *conds):
        assert len(conds) == self.n_cond, f"expected {self.n_cond} conditions"
        x = torch.cat([z, *conds], dim=1)              # (B, D*(1+n_cond), T)
        x = x.permute(0, 2, 1)                          # (B, T, D')
        x = self.in_proj(x)
        c = self.time_embed(t.unsqueeze(-1))
        for blk in self.blocks:
            x = blk(x, c)
        shift, scale = self.final_adaLN(c).chunk(2, dim=-1)
        x = self.final_norm(x) * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)
        x = self.out_proj(x)
        return x.permute(0, 2, 1)                       # (B, D, T)


@torch.no_grad()
def sample(model, conds, steps, device, seed=0):
    """Integrate the learned field from t=1 (noise) to t=0 (data), conditioned."""
    B = conds[0].shape[0]
    g = torch.Generator(device=device).manual_seed(seed)
    z = torch.randn(conds[0].shape, device=device, generator=g)
    ts = torch.linspace(1.0, 0.0, steps + 1, device=device)
    for i in range(steps):
        t = ts[i].expand(B)
        z = z + model(z, t, *conds) * (ts[i + 1] - ts[i])
    return z


@torch.no_grad()
def ae_decode(wrapper, z, orig_len):
    z_in = z.permute(0, 2, 1) if wrapper.model_type == "transformer" else z
    return wrapper.model.decode(z_in, original_length=orig_len)


def get(arr, idx, device):
    """Fetch a batch from either an in-RAM tensor or a read-only memmap."""
    if torch.is_tensor(arr):
        return arr[idx].to(device, non_blocking=True)
    return torch.from_numpy(np.array(arr[idx], dtype=np.float32, copy=True)).to(device)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--mixture_dir", required=True,
                   help="output dir of build_mixtures.py (must be built with the "
                        "same AE as --ae_checkpoint)")
    p.add_argument("--ae_checkpoint", required=True)
    p.add_argument("--run_name", required=True)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--epochs", type=int, default=150)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--embed_dim", type=int, default=384)
    p.add_argument("--depth", type=int, default=6)
    p.add_argument("--num_heads", type=int, default=8)
    p.add_argument("--max_train", type=int, default=0, help="0 = use all")
    p.add_argument("--data_on_gpu", action="store_true", default=True,
                   help="also move the latent tensors onto the GPU. The random "
                        "batch gather from a multi-GB CPU tensor is the dominant "
                        "per-step cost at short latents (measured 68.9 ms vs "
                        "0.14 ms per batch at (64,125)) -- 481x faster resident.")
    p.add_argument("--no_data_on_gpu", dest="data_on_gpu", action="store_false")
    p.add_argument("--in_memory", action="store_true", default=True,
                   help="load splits fully into RAM (24 GB for 50k mixtures); "
                        "predicted ~4x faster than random-accessing the memmap")
    p.add_argument("--no_in_memory", dest="in_memory", action="store_false")
    p.add_argument("--use_ot_coupling", action="store_true", default=True)
    p.add_argument("--no_ot_coupling", dest="use_ot_coupling", action="store_false")
    p.add_argument("--seed", type=int, default=42,
                   help="training seed: controls model init and the data/noise stream. "
                        "Needed for seed-replication; the earlier runs were unseeded "
                        "and are therefore not reproducible.")
    p.add_argument("--objective", choices=["rf", "mse"], default="rf",
                   help="'rf' = rectified flow (the model). 'mse' = a direct "
                        "deterministic regressor z_mix -> z_voice using the SAME "
                        "DiT body, trained with the same optimiser/schedule. A "
                        "regressor IS the conditional mean, so it bounds what the "
                        "'mix' task allows and separates 'the task is the limit' "
                        "from 'the RF/sampling path is the limit'.")
    p.add_argument("--cond", choices=["mix", "mix_other"], default="mix",
                   help="'mix' = realistic task, conditioned on z_mix only (no oracle, "
                        "so plain latent algebra cannot run); 'mix_other' = also given "
                        "z_other, which makes the target algebraically reachable on an "
                        "additive latent")
    p.add_argument("--eval_every", type=int, default=5)
    p.add_argument("--eval_clips", type=int, default=8)
    p.add_argument("--eval_seeds", type=int, default=4,
                   help="noise draws per eval.  An RF model SAMPLES from "
                        "p(z_voice|z_mix) but separation wants the conditional MEAN, "
                        "so the reported metrics average over draws: wavavg = mean of "
                        "decoded waveforms, latavg = decode of the mean latent, "
                        "bestofk = per-clip best draw (upper reference).")
    p.add_argument("--eval_seed_sweep", action="store_true", default=True)
    p.add_argument("--no_eval_seed_sweep", dest="eval_seed_sweep", action="store_false")
    p.add_argument("--sample_steps", type=int, default=32)
    p.add_argument("--step_sweep", default="1,2,4,8,16,32")
    p.add_argument("--val_batches", type=int, default=20,
                   help="val batches averaged per validation pass")
    p.add_argument("--output_dir", default="toy_training/checkpoints/fm_cond")
    p.add_argument("--wandb_project", default="toy_ae_phase")
    args = p.parse_args()

    device = torch.device(args.device)
    torch.manual_seed(args.seed)          # model init + data order + noise stream
    np.random.seed(args.seed)
    mdir = Path(args.mixture_dir)
    meta = json.loads((mdir / "meta.json").read_text())
    out = Path(args.output_dir); out.mkdir(parents=True, exist_ok=True)

    # ---- penalty for a mismatched dataset/AE pairing ----
    if Path(meta["ae_checkpoint"]).name != Path(args.ae_checkpoint).name:
        raise SystemExit(
            f"[FATAL] mixture_dir was built with {meta['ae_checkpoint']} but "
            f"--ae_checkpoint is {args.ae_checkpoint}. Latents live in the AE's "
            f"own space; rebuild the mixtures for this AE.")

    T_lat = meta["splits"]["train"]["latent_shape"][1]
    clip_samples = T_lat * 256                      # patch_size 256, 16 kHz

    # ---- data (memory-mapped) ----
    def load_split(name):
        s = meta["splits"][name]
        arrs = [np.load(s["files"][k], mmap_mode="r")
                for k in ("zmix", "zother", "zvoice")]
        if args.in_memory:
            t0 = time.time()
            arrs = [torch.from_numpy(np.array(a, dtype=np.float32)) for a in arrs]
            print(f"[Data] {name} loaded into RAM "
                  f"({sum(a.numel() for a in arrs) * 4 / 1e9:.1f} GB, "
                  f"{time.time()-t0:.0f}s)")
        return arrs

    tr_mix, tr_oth, tr_tgt = load_split("train")
    va_mix, va_oth, va_tgt = load_split("val")
    if args.max_train:
        tr_mix, tr_oth, tr_tgt = (a[:args.max_train] for a in (tr_mix, tr_oth, tr_tgt))
    n_train = len(tr_tgt)
    print(f"[Data] {mdir.name}: train {n_train}, val {len(va_tgt)}, "
          f"latent ({tr_tgt.shape[1]}, {T_lat}), AE "
          f"{Path(args.ae_checkpoint).stem}")
    if args.data_on_gpu and torch.is_tensor(tr_tgt):
        need = sum(a.numel() for a in (tr_mix, tr_oth, tr_tgt, va_mix, va_oth, va_tgt)) * 4
        free = torch.cuda.mem_get_info(device)[0]
        if need < 0.6 * free:
            tr_mix, tr_oth, tr_tgt = (a.to(device) for a in (tr_mix, tr_oth, tr_tgt))
            va_mix, va_oth, va_tgt = (a.to(device) for a in (va_mix, va_oth, va_tgt))
            print(f"[Data] latents resident on {device} "
                  f"({need/1e9:.1f} GB, {free/1e9:.1f} GB free)")
        else:
            print(f"[Data] keeping latents on CPU (need {need/1e9:.1f} GB, "
                  f"only {free/1e9:.1f} GB free on {device})")

    # ---- wandb ----
    wandb_run = None
    if args.wandb_project:
        import wandb
        wandb_run = wandb.init(project=args.wandb_project, name=args.run_name,
                               config={**vars(args), "mixture_dir": str(mdir),
                                       "n_train": n_train, "n_val": int(len(va_tgt)),
                                       "latent_shape": meta["splits"]["train"]["latent_shape"],
                                       "task": "conditional latent stem extraction "
                                               "(z_mix, z_other) -> z_voice"})

    # ---- model ----
    # the regressor consumes z_mix directly in the input slot, so it needs no
    # separate conditioning latent; the transformer body stays identical
    n_cond = 0 if args.objective == "mse" else (1 if args.cond == "mix" else 2)
    model = CondLatentDiT(channels=tr_tgt.shape[1], embed_dim=args.embed_dim,
                          depth=args.depth, num_heads=args.num_heads,
                          n_cond=n_cond).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[Model] CondLatentDiT {n_params/1e6:.2f}M params, depth={args.depth}, "
          f"dim={args.embed_dim}, heads={args.num_heads}, objective={args.objective}, "
          f"cond={args.cond} ({n_cond} conditioning latents)")
    if wandb_run:
        wandb_run.summary["n_params_M"] = n_params / 1e6

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    steps_per_epoch = max(1, n_train // args.batch_size)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=args.lr, total_steps=args.epochs * steps_per_epoch)

    best_ckpt = out / f"{args.run_name}_best.pt"
    final_ckpt = out / f"{args.run_name}_final.pt"

    # ---- eval assets: AE, raw voice wavs, and the fixed baseline ladder ----
    ae = ToyModelWrapper(args.ae_checkpoint, device)
    loss_fn = SAMELoss().to(device)
    K = min(args.eval_clips, len(va_tgt))
    eval_voice = np.load(mdir / "val_voice_wav.npy", mmap_mode="r")[:K]
    e_mix = get(va_mix, slice(0, K), device)
    e_oth = get(va_oth, slice(0, K), device)
    e_tgt = get(va_tgt, slice(0, K), device)

    def decode_wav(z):
        """Latent (K,D,T) -> waveform (K,T) numpy."""
        return ae_decode(ae, z, clip_samples).squeeze(1).cpu().numpy()

    def _sisdr_of(wav):
        return np.array([si_sdr_db(eval_voice[i], wav[i]) for i in range(len(wav))])

    def decode_stats(z):
        wav = decode_wav(z)
        sisdr = _sisdr_of(wav)
        ms = []
        for i in range(len(wav)):
            pr = torch.from_numpy(wav[i]).float().view(1, 1, -1).to(device)
            tg = torch.from_numpy(np.array(eval_voice[i], dtype=np.float32)).to(device).view(1, 1, -1)
            ms.append(loss_fn(pr, tg, use_phase=False).item())
        return float(np.mean(sisdr)), float(np.mean(ms))

    with torch.no_grad():
        base = {}
        base["ceiling"] = decode_stats(e_tgt)
        base["noop"] = decode_stats(e_mix)
        base["oracle"] = decode_stats(e_mix - e_oth)
    print(f"[Eval ladder] ceiling SI-SDR {base['ceiling'][0]:.2f} dB | "
          f"oracle {base['oracle'][0]:.2f} dB | no-op {base['noop'][0]:.2f} dB  "
          f"(K={K} val clips)")
    if wandb_run:
        for k, (s, m) in base.items():
            wandb_run.summary[f"baseline_{k}_sisdr"] = s
            wandb_run.summary[f"baseline_{k}_msstft"] = m

    step_list = [int(s) for s in args.step_sweep.split(",")]
    e_conds = (e_mix,) if n_cond == 1 else (e_mix, e_oth)

    def generate(zmix, steps, seed=0):
        """One forward pass for the regressor; RF sampling otherwise."""
        if args.objective == "mse":
            return model(zmix, torch.zeros(zmix.shape[0], device=device))
        return sample(model, (zmix,), steps, device, seed=seed)

    def run_eval(epoch):
        model.eval()
        ns = max(1, args.eval_seeds)
        with torch.no_grad():
            out_d = {}
            zs, wavs = [], []
            if args.objective == "mse":
                ns = 1        # deterministic: no sampling variance to average
            for sd in range(ns):
                z = generate(e_mix, args.sample_steps, seed=sd)
                zs.append(z)
                wavs.append(decode_wav(z))
            per_clip = np.stack([_sisdr_of(w) for w in wavs])       # (ns, K)
            wav_mean = np.mean(wavs, axis=0)
            out_d["eval/sisdr_fm_seed0"] = float(per_clip[0].mean())
            out_d["eval/sisdr_fm_wavavg"] = float(_sisdr_of(wav_mean).mean())
            out_d["eval/sisdr_fm_bestofk"] = float(per_clip.max(axis=0).mean())
            z_mean = torch.stack(zs, 0).mean(0)
            sm, mm = decode_stats(z_mean)
            out_d["eval/sisdr_fm_latavg"] = sm
            out_d["eval/msstft_fm_latavg"] = mm
            # headline = waveform-average over draws: the MMSE estimator under SI-SDR
            out_d["eval/msstft_fm"] = mm
            out_d["eval/sisdr_fm"] = out_d["eval/sisdr_fm_wavavg"]
            # step sweep on a subset: the step-count curve is about shape, and this
            # keeps eval overhead off the critical path
            nsw = min(8, K)
            sw_conds = tuple(c[:nsw] for c in e_conds)
            for st in step_list:
                w_sw = decode_wav(generate(sw_conds[0], st, seed=0))
                out_d[f"eval/sisdr_steps_{st}"] = float(_sisdr_of(w_sw).mean())
            rel = ((z_mean - e_tgt).flatten(1).norm(dim=1) /
                   (e_tgt.flatten(1).norm(dim=1) + 1e-12))
            out_d["eval/latent_rel_err_fm"] = float(rel.mean())
            for k, (bs, _) in base.items():
                out_d[f"eval/sisdr_ref_{k}"] = bs
        model.train()
        print(f"    [eval@{epoch}] FM wavavg {out_d['eval/sisdr_fm_wavavg']:.2f} | "
              f"seed0 {out_d['eval/sisdr_fm_seed0']:.2f} | "
              f"latavg {out_d['eval/sisdr_fm_latavg']:.2f} | "
              f"bestof{ns} {out_d['eval/sisdr_fm_bestofk']:.2f} | "
              f"no-op {base['noop'][0]:.2f} | oracle {base['oracle'][0]:.2f} | "
              f"ceiling {base['ceiling'][0]:.2f} | sweep "
              + " ".join(f"{st}:{out_d[f'eval/sisdr_steps_{st}']:.2f}"
                         for st in step_list))
        return out_d

    # ---- training ----
    log = []
    best_val = float("inf")
    t0 = time.time()
    for epoch in range(1, args.epochs + 1):
        model.train()
        perm = torch.randperm(n_train)
        ep_loss, nb = 0.0, 0
        for i in range(0, n_train - args.batch_size + 1, args.batch_size):
            idx = perm[i:i + args.batch_size]
            z_data = get(tr_tgt, idx, device)
            zmix = get(tr_mix, idx, device)
            zoth = get(tr_oth, idx, device)
            B = z_data.shape[0]
            if args.objective == "mse":
                # deterministic regression: one forward from z_mix, no noise, no time
                pred = model(zmix, torch.zeros(B, device=device))
                loss = F.mse_loss(pred, z_data)
            else:
                t = torch.rand(B, device=device)
                noise = torch.randn_like(z_data)
                if args.use_ot_coupling and B > 1:
                    noise = ot_coupling(z_data, noise)
                z_t = (1 - t).view(B, 1, 1) * z_data + t.view(B, 1, 1) * noise
                v_target = noise - z_data
                v_pred = model(z_t, t, *((zmix,) if n_cond == 1 else (zmix, zoth)))
                loss = F.mse_loss(v_pred, v_target)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            ep_loss += loss.item(); nb += 1
            if wandb_run and nb % 50 == 0:
                wandb_run.log({"fm/batch_loss": loss.item(),
                               "fm/batch_lr": scheduler.get_last_lr()[0],
                               "step": (epoch - 1) * steps_per_epoch + nb})
        train_loss = ep_loss / max(nb, 1)

        # ---- validation on held-out mixtures ----
        model.eval()
        with torch.no_grad():
            vl, vn = 0.0, 0
            for i in range(0, min(len(va_tgt), args.val_batches * args.batch_size),
                           args.batch_size):
                z_data = get(va_tgt, slice(i, i + args.batch_size), device)
                zmix = get(va_mix, slice(i, i + args.batch_size), device)
                zoth = get(va_oth, slice(i, i + args.batch_size), device)
                B = z_data.shape[0]
                if args.objective == "mse":
                    vl += F.mse_loss(model(zmix, torch.zeros(B, device=device)),
                                     z_data).item()
                    vn += 1
                    continue
                acc = 0.0
                for _ in range(4):
                    t = torch.rand(B, device=device)
                    noise = torch.randn_like(z_data)
                    if args.use_ot_coupling and B > 1:
                        noise = ot_coupling(z_data, noise)
                    z_t = (1 - t).view(B, 1, 1) * z_data + t.view(B, 1, 1) * noise
                    acc += F.mse_loss(
                        model(z_t, t, *((zmix,) if n_cond == 1 else (zmix, zoth))),
                        noise - z_data).item()
                vl += acc / 4; vn += 1
            val_loss = vl / max(vn, 1)
        model.train()

        star = ""
        if val_loss < best_val:
            best_val = val_loss
            torch.save({"model_state_dict": model.state_dict(),
                        "channels": int(tr_tgt.shape[1]), "t_lat": int(T_lat),
                        "embed_dim": args.embed_dim, "depth": args.depth,
                        "num_heads": args.num_heads, "n_cond": n_cond,
                        "cond": args.cond, "objective": args.objective,
                        "seed": args.seed,
                        "ae_checkpoint": args.ae_checkpoint,
                        "mixture_dir": str(mdir),
                        "epoch": epoch, "val_loss": val_loss,
                        "task": "cond_stem_extraction"}, best_ckpt)
            star = " *"

        row = {"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss}
        log.append(row)
        msg = (f"[Epoch {epoch:4d}/{args.epochs}] train={train_loss:.6f} "
               f"val={val_loss:.6f}{star}  ({(time.time()-t0)/60:.1f} min)")
        if wandb_run:
            wandb_run.log({"fm/train_loss": train_loss, "fm/val_loss": val_loss,
                           "fm/best_val": best_val, "fm/lr": scheduler.get_last_lr()[0],
                           "epoch": epoch}, step=(epoch * steps_per_epoch))
        if args.eval_every and (epoch % args.eval_every == 0 or epoch == args.epochs):
            ed = run_eval(epoch)
            row.update(ed)
            if wandb_run:
                wandb_run.log(ed, step=(epoch * steps_per_epoch))
        print(msg)

    torch.save({"model_state_dict": model.state_dict(),
                "channels": int(tr_tgt.shape[1]), "t_lat": int(T_lat),
                "embed_dim": args.embed_dim, "depth": args.depth,
                "num_heads": args.num_heads, "n_cond": n_cond,
                "cond": args.cond, "objective": args.objective,
                "seed": args.seed,
                "ae_checkpoint": args.ae_checkpoint, "mixture_dir": str(mdir),
                "epoch": args.epochs, "val_loss": val_loss,
                "task": "cond_stem_extraction"}, final_ckpt)

    results = {"run_name": args.run_name, "ae_checkpoint": args.ae_checkpoint,
               "mixture_dir": str(mdir), "n_train": n_train,
               "n_val": int(len(va_tgt)), "n_params_M": n_params / 1e6,
               "latent_shape": meta["splits"]["train"]["latent_shape"],
               "baselines": {k: {"sisdr": v[0], "msstft": v[1]} for k, v in base.items()},
               "eval_clips": K, "sample_steps": args.sample_steps,
               "step_sweep": step_list, "best_val": best_val,
               "loss_curve": log, "args": vars(args)}
    (out / f"{args.run_name}_results.json").write_text(json.dumps(results, indent=2))
    print(f"\nSaved: {best_ckpt}\nSaved: {final_ckpt}\n"
          f"Saved: {out / f'{args.run_name}_results.json'}")
    print(f"Best val loss: {best_val:.6f}   ({time.time()-t0:.0f}s total)")
    if wandb_run:
        wandb_run.summary["best_val"] = best_val
        wandb_run.finish()


if __name__ == "__main__":
    main()
