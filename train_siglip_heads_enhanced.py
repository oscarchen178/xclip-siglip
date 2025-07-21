#!/usr/bin/env python
"""
Train enhanced 2-layer projection heads on precomputed image/text embeddings
using a SigLIP-style (pairwise sigmoid) contrastive objective with optional
label smoothing, balancing, and positive weighting.

Inputs (default: ./outputs):
  image_embeddings.pt          [N, D_img]
  text_embeddings.pt           [N, D_txt]

Outputs:
  proj_image.pt, proj_text.pt          (latest)
  best_state.pt                        (best by avg Recall@10)
  final_state.pt
  training_log.json
  proj_image_embeddings.pt / proj_text_embeddings.pt (final projected)
  retrieval_metrics_epoch_*.json

Usage (basic):
  python train_siglip_heads_enhanced.py

Example (with smoothing, different proj dim):
  python train_siglip_heads_enhanced.py --label_smoothing 0.05 --proj_dim 512
"""

from __future__ import annotations
import json
import math
import time
import argparse
from pathlib import Path
from typing import Optional, Dict, Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

# ----------------------------- Dataset -------------------------------------- #
class PairEmbeddingDataset(Dataset):
    def __init__(self, img_emb: torch.Tensor, txt_emb: torch.Tensor):
        assert img_emb.shape[0] == txt_emb.shape[0], "Image/Text count mismatch"
        self.img = img_emb
        self.txt = txt_emb

    def __len__(self):
        return self.img.shape[0]

    def __getitem__(self, idx):
        return self.img[idx], self.txt[idx]

# ----------------------------- Projection Heads ------------------------------ #
class ProjectionHead(nn.Module):
    def __init__(self, in_dim: int, out_dim: int,
                 hidden_dim: Optional[int] = None,
                 dropout: float = 0.1,
                 use_layernorm: bool = True):
        super().__init__()
        h = hidden_dim or max(in_dim, out_dim)
        self.net = nn.Sequential(
            nn.Linear(in_dim, h),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(h, out_dim)
        )
        self.norm = nn.LayerNorm(out_dim) if use_layernorm else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.net(x)
        z = self.norm(z)
        z = F.normalize(z, p=2, dim=-1)
        return z

class DualProjectionModel(nn.Module):
    def __init__(self, d_img: int, d_txt: int, proj_dim: int,
                 hidden_mult: float = 1.0, dropout: float = 0.1):
        super().__init__()
        h_img = int(max(d_img, proj_dim) * hidden_mult)
        h_txt = int(max(d_txt, proj_dim) * hidden_mult)
        self.proj_img = ProjectionHead(d_img, proj_dim, h_img, dropout)
        self.proj_txt = ProjectionHead(d_txt, proj_dim, h_txt, dropout)
        self.log_tau = nn.Parameter(torch.log(torch.tensor(0.05)))  # tau init ~0.05

    def forward(self, img_feats: torch.Tensor, txt_feats: torch.Tensor):
        z_i = self.proj_img(img_feats)
        z_t = self.proj_txt(txt_feats)
        tau = self.log_tau.exp()
        return z_i, z_t, tau

# ----------------------------- SigLIP Loss ---------------------------------- #
def siglip_loss(
    z_i: torch.Tensor,
    z_t: torch.Tensor,
    tau: torch.Tensor,
    *,
    label_smoothing: float = 0.0,
    balance: bool = True,
    use_pos_weight: bool = False
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Pairwise sigmoid (SigLIP) contrastive loss with enhancements.

    Args:
        z_i, z_t: [B, D] normalized embeddings
        tau:     scalar temperature
        label_smoothing: 0.0 => no smoothing; else e.g. 0.05
        balance: if True, average positive and negative losses separately
        use_pos_weight: if True, weight positives by (B-1) instead of balance
                        (ignored if balance=True)

    Returns:
        loss, similarity_matrix
    """
    sim = (z_i @ z_t.T) / tau  # [B, B]
    B = sim.size(0)
    eye = torch.eye(B, device=sim.device, dtype=sim.dtype)

    # Targets with optional label smoothing
    if label_smoothing > 0:
        pos_val = 1.0 - label_smoothing
        neg_val = label_smoothing
    else:
        pos_val, neg_val = 1.0, 0.0

    target = eye * pos_val + (1 - eye) * neg_val  # [B,B]

    if balance:
        # Separate positive / negative sets; average them independently
        pos_mask = eye.bool()
        neg_mask = ~pos_mask
        pos_logits = sim[pos_mask]
        neg_logits = sim[neg_mask]

        # Log-sigmoid forms
        log_sigma_pos = -F.softplus(-pos_logits)                       # log σ
        log_one_minus_sigma_neg = -neg_logits - F.softplus(-neg_logits) # log (1-σ)

        if label_smoothing > 0:
            # For positives y=pos_val: y log σ + (1-y) log (1-σ)
            log_one_minus_sigma_pos = -pos_logits - F.softplus(-pos_logits)
            pos_loss = -(pos_val * log_sigma_pos +
                         (1 - pos_val) * log_one_minus_sigma_pos)
            # For negatives y=neg_val
            log_sigma_neg = -F.softplus(-neg_logits)
            neg_loss = -(neg_val * log_sigma_neg +
                         (1 - neg_val) * log_one_minus_sigma_neg)
        else:
            pos_loss = -log_sigma_pos
            neg_loss = -log_one_minus_sigma_neg

        loss = 0.5 * (pos_loss.mean() + neg_loss.mean())

    else:
        # Unbalanced BCE; optionally apply pos_weight
        if use_pos_weight:
            pos_weight = torch.tensor([B - 1], device=sim.device, dtype=sim.dtype)
            loss = F.binary_cross_entropy_with_logits(sim, eye, pos_weight=pos_weight)
        else:
            loss = F.binary_cross_entropy_with_logits(sim, eye)
    return loss, sim.detach()

# ----------------------------- Scheduler ------------------------------------ #
def build_scheduler(optimizer, total_steps, warmup_steps, base_lr, min_lr):
    # warmup_steps = min(warmup_steps, max(0, total_steps - 1))
    warmup_steps = min(5, total_steps // 10)
    def lr_lambda(step: int):
        if warmup_steps > 0 and step < warmup_steps:
            return float(step) / float(max(1, warmup_steps))
        progress = (step - warmup_steps) / max(1, (total_steps - warmup_steps))
        cosine = 0.5 * (1 + math.cos(math.pi * progress))
        scale = cosine * (1 - min_lr / base_lr) + (min_lr / base_lr)
        return scale
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

# ----------------------------- Retrieval Metrics ---------------------------- #
@torch.no_grad()
def retrieval_metrics(z_img: torch.Tensor, z_txt: torch.Tensor, ks=(1,5,10)) -> Dict[str, float]:
    """
    Compute image->text and text->image Recall@K for given projected & normalized embeddings.
    """
    sim = z_img @ z_txt.T  # [N,N]
    N = sim.size(0)
    # Image -> Text
    ranks_it = sim.argsort(dim=1, descending=True)
    diag = torch.arange(N, device=sim.device)
    pos_ranks_it = (ranks_it == diag.unsqueeze(1)).nonzero()[:, 1]
    # Text -> Image
    ranks_ti = sim.T.argsort(dim=1, descending=True)
    pos_ranks_ti = (ranks_ti == diag.unsqueeze(1)).nonzero()[:, 1]

    metrics = {}
    for k in ks:
        metrics[f"i2t_R@{k}"] = (pos_ranks_it < k).float().mean().item()
        metrics[f"t2i_R@{k}"] = (pos_ranks_ti < k).float().mean().item()
    metrics["mean_R@10"] = 0.5 * (metrics.get("i2t_R@10", 0) + metrics.get("t2i_R@10", 0))
    return metrics

def to_jsonable(obj):
    if isinstance(obj, (str, int, float, bool)) or obj is None:
        return obj
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, (list, tuple)):
        return [to_jsonable(x) for x in obj]
    if isinstance(obj, dict):
        return {k: to_jsonable(v) for k, v in obj.items()}
    return str(obj)  # fallback


# ----------------------------- Training Loop -------------------------------- #
def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    img_emb = torch.load(args.image_embeddings, map_location="cpu").float()
    txt_emb = torch.load(args.text_embeddings, map_location="cpu").float()
    N, d_img = img_emb.shape
    _, d_txt = txt_emb.shape
    print(f"Loaded embeddings: N={N} d_img={d_img} d_txt={d_txt}")

    dataset = PairEmbeddingDataset(img_emb, txt_emb)
    batch_size = min(args.batch_size, N)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=False)

    model = DualProjectionModel(d_img, d_txt, args.proj_dim,
                                hidden_mult=args.hidden_mult,
                                dropout=args.dropout).to(device)

    optimizer = torch.optim.AdamW(model.parameters(),
                                  lr=args.lr,
                                  betas=(0.9, 0.999),
                                  weight_decay=args.weight_decay)

    total_steps = args.epochs * len(loader)
    if args.warmup_frac is not None:
        args.warmup_steps = max(1, int(args.warmup_frac * total_steps))
    scheduler = build_scheduler(optimizer, total_steps, args.warmup_steps, args.lr, args.min_lr)

    scaler = torch.cuda.amp.GradScaler(enabled=args.amp)

    config_json = to_jsonable(vars(args))
    log = {"config": config_json, "history": []}

    best_metric = -1.0
    patience = 5          # <-- you can make this an argparse arg
    no_improve = 0
    global_step = 0
    t_start = time.time()
    if args.run_name:
        args.output_dir = args.output_dir / args.run_name
        args.output_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_loss = 0.0
        for img_batch, txt_batch in loader:
            img_batch = img_batch.to(device, non_blocking=True)
            txt_batch = txt_batch.to(device, non_blocking=True)

            with torch.cuda.amp.autocast(enabled=args.amp):
                z_i, z_t, tau = model(img_batch, txt_batch)
                loss, _ = siglip_loss(
                    z_i, z_t, tau,
                    label_smoothing=args.label_smoothing,
                    balance=not args.use_pos_weight_balancing,
                    use_pos_weight=args.use_pos_weight_balancing
                )

            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            if args.grad_clip is not None:
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            epoch_loss += loss.item()
            if global_step % args.log_interval == 0:
                print(f"[E{epoch:02d} S{global_step:05d}] "
                      f"loss={loss.item():.4f} tau={tau.item():.4f} "
                      f"lr={scheduler.get_last_lr()[0]:.3e}")
            global_step += 1

        avg_loss = epoch_loss / len(loader)

        # Evaluation (on full set)
        model.eval()
        with torch.no_grad():
            z_img_all, z_txt_all, _ = model(img_emb.to(device), txt_emb.to(device))
        metrics = retrieval_metrics(z_img_all, z_txt_all)
        metrics["avg_train_loss"] = avg_loss
        metrics["tau"] = model.log_tau.exp().item()
        log["history"].append({"epoch": epoch, **metrics})

        # Save per-epoch metrics
        with open(args.output_dir / f"retrieval_metrics_epoch_{epoch}.json", "w") as f:
            json.dump(metrics, f, indent=2)

        # Track best
        if metrics["mean_R@10"] > best_metric:
            best_metric = metrics["mean_R@10"]
            no_improve = 0
            torch.save(model.state_dict(), args.output_dir / "best_state.pt")
        else:
            no_improve += 1
        if no_improve >= patience:
            print("Early stopping.")
        
            break

        # Regular checkpoint
        if epoch % args.save_interval == 0 or epoch == args.epochs:
            torch.save(model.proj_img.state_dict(), args.output_dir / "proj_image.pt")
            torch.save(model.proj_txt.state_dict(), args.output_dir / "proj_text.pt")
            torch.save(model.state_dict(), args.output_dir / "final_state.pt")
            with open(args.output_dir / "training_log.json", "w") as f:
                json.dump(log, f, indent=2)

        print(f"Epoch {epoch} | loss={avg_loss:.4f} "
              f"i2t@10={metrics['i2t_R@10']:.3f} t2i@10={metrics['t2i_R@10']:.3f} "
              f"tau={metrics['tau']:.4f}")

    # Final projected embeddings (from final state)
    with torch.no_grad():
        z_img_all, z_txt_all, _ = model(img_emb.to(device), txt_emb.to(device))
    torch.save(z_img_all.cpu(), args.output_dir / "proj_image_embeddings.pt")
    torch.save(z_txt_all.cpu(), args.output_dir / "proj_text_embeddings.pt")

    # Save final log
    with open(args.output_dir / "training_log.json", "w") as f:
        json.dump(log, f, indent=2)

    print(f"Training done in {(time.time()-t_start):.1f}s. Best mean R@10={best_metric:.3f}")

# ----------------------------- CLI ------------------------------------------ #
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--image_embeddings", type=Path, default=Path("outputs/image_embeddings.pt"))
    p.add_argument("--text_embeddings", type=Path, default=Path("outputs/text_embeddings.pt"))
    p.add_argument("--output_dir", type=Path, default=Path("outputs"))
    p.add_argument("--proj_dim", type=int, default=512)
    p.add_argument("--hidden_mult", type=float, default=1.0)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--min_lr", type=float, default=1e-5)
    p.add_argument("--warmup_steps", type=int, default=1000)
    p.add_argument("--weight_decay", type=float, default=1e-2)
    p.add_argument("--label_smoothing", type=float, default=0.0,
                   help="e.g. 0.05 for smoothed targets")
    p.add_argument("--use_pos_weight_balancing", action="store_true",
                   help="Use BCE pos_weight=B-1 instead of balanced averaging.")
    p.add_argument("--grad_clip", type=float, default=1.0)
    p.add_argument("--log_interval", type=int, default=25)
    p.add_argument("--save_interval", type=int, default=5)
    p.add_argument("--amp", action="store_true", help="Enable mixed precision.")
    p.add_argument("--run_name", type=str, default="", help="Name suffix for this run.")
    p.add_argument("--warmup_frac", type=float, default=None,
               help="If set, overrides warmup_steps = int(warmup_frac * total_steps).")

    return p.parse_args()

if __name__ == "__main__":
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    train(args)