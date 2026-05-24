from datetime import datetime
from pathlib import Path

import jax
import jax.numpy as jnp
from omegaconf import DictConfig, OmegaConf


def create_experiment_folder(save_name: str, root: Path) -> Path:
    """Create unique fodler-name where to save yaml and expeiremnt"""
    root.mkdir(parents=True, exist_ok=True)
    base_path = root / save_name
    idx = 0
    while True:
        exp_path = base_path if idx == 0 else root / f"{save_name}_{idx}"
        try:
            print("create")
            exp_path.mkdir()
            return exp_path
        except FileExistsError:
            print("Hincrement")
            idx += 1


def save_config(config: dict, save_path_exp: Path) -> None:
    # --- Check if folder-nmae is still free, if so, create folder
    save_path_yml = save_path_exp / "config"
    save_path_yml.mkdir()
    OmegaConf.save(config, save_path_yml / "config.yaml")


def create_timing_name(base_path: Path, save_name: str) -> Path:
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    save_path_exp = base_path / f"{save_name}_{timestamp}"
    save_path_exp.mkdir(parents=True)
    return save_path_exp
