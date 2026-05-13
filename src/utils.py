# src/utils.py
import torch
import numpy as np
import random
from pathlib import Path


def set_seed(seed: int = 42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False


class CheckpointManager:
    """
    Saves and loads complete training state.
    Maintains two files per run:
      latest.pt — overwritten every epoch, safe resume after disconnect
      best.pt   — best validation loss so far
    """

    def __init__(self, save_dir: Path, run_name: str):
        self.save_dir  = Path(save_dir) / run_name
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.best_loss = float("inf")

    def save(self, epoch: int, model: torch.nn.Module,
             optimizer, scheduler, scaler,
             metrics: dict, config: dict,
             wandb_run_id: str = None):

        state = {
            "epoch":        epoch,
            "model":        model.state_dict(),
            "optimizer":    optimizer.state_dict(),
            "scheduler":    scheduler.state_dict() if scheduler else None,
            "scaler":       scaler.state_dict()    if scaler    else None,
            "metrics":      metrics,
            "config":       config,
            "wandb_run_id": wandb_run_id,
        }
        torch.save(state, self.save_dir / "latest.pt")

        val_loss = metrics.get("val_loss", float("inf"))
        if val_loss < self.best_loss:
            self.best_loss = val_loss
            torch.save(state, self.save_dir / "best.pt")
            print(f"  ★ New best saved (val_loss {self.best_loss:.4f})")

    def load(self, model: torch.nn.Module,
             optimizer=None, scheduler=None, scaler=None,
             checkpoint: str = "latest") -> dict:
        path = self.save_dir / f"{checkpoint}.pt"
        if not path.exists():
            print(f"No checkpoint at {path}. Starting from scratch.")
            return {"epoch": 0, "wandb_run_id": None}

        # weights_only=False needed to load optimizer/scheduler states
        # safe here since we only load our own checkpoints
        state = torch.load(path, map_location="cpu", weights_only=False)
        model.load_state_dict(state["model"])
        if optimizer and state.get("optimizer"):
            optimizer.load_state_dict(state["optimizer"])
        if scheduler and state.get("scheduler"):
            scheduler.load_state_dict(state["scheduler"])
        if scaler and state.get("scaler"):
            scaler.load_state_dict(state["scaler"])
        print(f"Resumed from epoch {state['epoch']} "
              f"| metrics: {state.get('metrics', {})}")
        return state

    def exists(self, checkpoint: str = "latest") -> bool:
        return (self.save_dir / f"{checkpoint}.pt").exists()