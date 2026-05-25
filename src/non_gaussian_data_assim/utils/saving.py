from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import jax.numpy as jnp
import numpy as np
from omegaconf import DictConfig, OmegaConf


@dataclass(frozen=True)
class ExperimentSaver:
    root: Path
    name: str
    path: Path

    @classmethod
    def create(
        cls, save_name: str, root: Path = Path("../experiments")
    ) -> "ExperimentSaver":
        """This is like an overload in c++, instantiate a class given a exp_name"""
        save_name = clean_exp_name(save_name)
        root.mkdir(parents=True, exist_ok=True)
        exp_path = create_unique_folder(root=root, save_name=save_name)
        return cls(root=root, name=save_name, path=exp_path)

    def subdir(self, name: str) -> Path:
        path = self.path / name
        path.mkdir(parents=True, exist_ok=True)
        return path

    def save_config(self, config: DictConfig | dict[str, Any]) -> Path:
        config_dir = self.subdir("config")
        config_path = config_dir / "config.yaml"
        OmegaConf.save(config, config_path)
        return config_path

    def save_array(self, name: str, array: jnp.ndarray) -> Path:
        arrays_dir = self.subdir("data")
        array_path = arrays_dir / f"{name}.npy"
        jnp.save(array_path, array, allow_pickle=False)
        return array_path

    def save_metrics(
        self, filename: str, metrics: dict[str, Any], save_summary: bool = True
    ) -> Path:
        metrics_dir = self.subdir("metrics")
        path = metrics_dir / f"{filename}.npz"

        metrics = {key: np.asarray(value) for key, value in metrics.items()}

        np.savez(path, allow_pickle=False, **metrics)

        if save_summary:
            summary = {
                key: float(value) for key, value in metrics.items() if value.ndim == 0
            }
            OmegaConf.save(
                OmegaConf.create(summary),
                metrics_dir / f"{filename}_summary.yaml",
            )
        return path


def clean_exp_name(save_name: str) -> str:
    save_name = save_name.strip()
    if not save_name:
        raise ValueError(
            "Error, empty experiment name was passed. Specify either a name or set to false in config file."
        )
    return save_name


def create_unique_folder(root: Path, save_name: str) -> Path:
    base_path = root / save_name
    idx = 0
    while True:
        exp_path = base_path if idx == 0 else root / f"{save_name}_{idx}"
        try:
            exp_path.mkdir()
            return exp_path
        except FileExistsError:
            idx += 1
