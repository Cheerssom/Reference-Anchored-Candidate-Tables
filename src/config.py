"""Centralized configuration — replaces hardcoded paths.

Usage:
    from config import CFG
    data_path = CFG.data_root / "cifar-100-python"
"""
import os
from pathlib import Path


class Config:
    def __init__(self):
        # Data roots (override via environment variables)
        self.data_root = Path(os.environ.get("JLT_DATA",
                                             Path(__file__).resolve().parents[1] / "data"))
        self.baseline_dir = Path(os.environ.get(
            "JLT_BASELINE", Path(__file__).resolve().parents[1] / "baselines"))
        self.output_dir = Path(os.environ.get("JLT_OUTPUT",
                                              Path(__file__).resolve().parents[1]))
        self.device = os.environ.get("JLT_DEVICE",
                                     "cuda" if torch_cuda() else "cpu")

    @property
    def cifar_root(self):
        return self.data_root / "cifar-100-python"

    @property
    def runs_dir(self):
        return self.output_dir / "runs"

    @property
    def cache_dir(self):
        return self.output_dir / "cache"

    @property
    def results_dir(self):
        return self.output_dir / "experiments"


def torch_cuda():
    try:
        import torch
        return torch.cuda.is_available()
    except ImportError:
        return False


CFG = Config()
