# src/train.py
import time
import torch
import torch.nn as nn
from pathlib import Path
from src.losses import WeightedBCELoss, CombinedLoss
from src.metrics import compute_all_metrics
from src.utils import CheckpointManager


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ── optimizer factory ─────────────────────────────────────────────────────────

def build_optimizer(model: nn.Module, config: dict):
    name = config["training"]["optimizer"]
    lr   = float(config["training"]["lr"])
    if name == "Adam":
        return torch.optim.Adam(model.parameters(), lr=lr)
    elif name == "AdamW":
        return torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    elif name == "SGD":
        return torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    raise ValueError(f"Unknown optimizer: {name}")


# ── scheduler factory ─────────────────────────────────────────────────────────

def build_scheduler(optimizer, config: dict):
    name   = config["training"]["scheduler"]
    epochs = config["training"]["epochs"]
    if name == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=epochs, eta_min=1e-7)
    elif name == "plateau":
        return torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.1, patience=10, min_lr=1e-7)
    elif name == "none":
        return None
    raise ValueError(f"Unknown scheduler: {name}")


# ── loss factory ──────────────────────────────────────────────────────────────

def build_loss(config: dict):
    name = config["training"].get("loss", "weighted_bce")
    if name == "weighted_bce":
        return WeightedBCELoss()
    elif name == "combined":
        alpha = config["training"].get("loss_alpha", 0.8)
        return CombinedLoss(alpha=alpha)
    raise ValueError(f"Unknown loss: {name}")


# ── single epoch ──────────────────────────────────────────────────────────────

def train_one_epoch(model, loader, optimizer, loss_fn,
                    scaler, device) -> dict:
    model.train()
    total_loss = 0.0

    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()

        with torch.amp.autocast('cuda'):
            logits = model(x)
            loss   = loss_fn(logits, y)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()

    n = len(loader)
    return {"train_loss": total_loss / n}


@torch.no_grad()
def validate(model, loader, loss_fn, device) -> dict:
    model.eval()
    total_loss = 0.0
    metrics    = {"ssim": 0.0, "pearson": 0.0, "psnr": 0.0}

    for x, y in loader:
        x, y   = x.to(device), y.to(device)
        with torch.amp.autocast('cuda'):
            logits = model(x)
            loss   = loss_fn(logits, y)
        total_loss += loss.item()

        batch_metrics = compute_all_metrics(logits, y)
        for k in metrics:
            metrics[k] += batch_metrics[k]

    n = len(loader)
    metrics["val_loss"] = total_loss / n
    for k in ["ssim", "pearson", "psnr"]:
        metrics[k] /= n
    return metrics


# ── full training loop ────────────────────────────────────────────────────────

def train(model, train_loader, val_loader, config: dict,
          checkpoint_dir: Path, run_name: str,
          wandb_run=None, resume: bool = True):

    model     = model.to(DEVICE)
    optimizer = build_optimizer(model, config)
    scheduler = build_scheduler(optimizer, config)
    loss_fn   = build_loss(config)
    scaler    = torch.amp.GradScaler('cuda')
    ckpt      = CheckpointManager(checkpoint_dir, run_name)

    start_epoch  = 0
    wandb_run_id = wandb_run.id if wandb_run else None

    # resume if checkpoint exists
    if resume and ckpt.exists("latest"):
        state        = ckpt.load(model, optimizer, scheduler, scaler)
        start_epoch  = state["epoch"] + 1
        wandb_run_id = state.get("wandb_run_id", wandb_run_id)
        print(f"Resuming from epoch {start_epoch}")

    epochs        = config["training"]["epochs"]
    log_interval  = config["logging"]["log_interval"]
    patience      = config["training"].get("early_stopping_patience", 20)
    best_val_loss = float("inf")
    no_improve    = 0

    print(f"Training on {DEVICE} — {epochs} epochs")
    t0 = time.time()

    for epoch in range(start_epoch, epochs):

        train_metrics = train_one_epoch(
            model, train_loader, optimizer, loss_fn, scaler, DEVICE)
        val_metrics   = validate(
            model, val_loader, loss_fn, DEVICE)

        # step scheduler
        if scheduler:
            if isinstance(scheduler,
                          torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(val_metrics["val_loss"])
            else:
                scheduler.step()

        all_metrics = {**train_metrics, **val_metrics,
                       "lr": optimizer.param_groups[0]["lr"],
                       "epoch": epoch}

        # logging
        if (epoch + 1) % log_interval == 0 or epoch == 0:
            elapsed = (time.time() - t0) / 60
            print(f"Epoch {epoch+1:04}/{epochs} | "
                  f"train_loss: {train_metrics['train_loss']:.4f} | "
                  f"val_loss: {val_metrics['val_loss']:.4f} | "
                  f"pearson: {val_metrics['pearson']:.4f} | "
                  f"ssim: {val_metrics['ssim']:.4f} | "
                  f"psnr: {val_metrics['psnr']:.4f} | "
                  f"lr: {all_metrics['lr']:.2e} | "
                  f"{elapsed:.1f}min elapsed")

        if wandb_run:
            wandb_run.log(all_metrics)

        # checkpoint
        ckpt.save(epoch, model, optimizer, scheduler, scaler,
                  all_metrics, config, wandb_run_id)

        # early stopping
        val_loss = val_metrics["val_loss"]
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            no_improve    = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                print(f"Early stopping at epoch {epoch+1} "
                      f"(no improvement for {patience} epochs)")
                break

    elapsed = (time.time() - t0) / 60
    print(f"Done — {elapsed:.1f} min total")
    return model