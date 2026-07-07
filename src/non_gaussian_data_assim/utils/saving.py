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
    def create(cls, save_name: str, root: Path) -> "ExperimentSaver":
        """This is like an overload in c++, instantiate a class given a exp_name"""

        root.mkdir(parents=True, exist_ok=True)
        exp_path = create_unique_folder(root=root, save_name=save_name)
        print(f"Directory made: {exp_path}")
        return cls(root=root, name=save_name, path=exp_path)

    def subdir(self, name: str) -> Path:
        path = self.path / name
        path.mkdir(parents=True, exist_ok=True)
        return path

    def save_config(self, config: DictConfig) -> Path:
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
        self, name: str, metrics: dict[str, Any], save_summary: bool = True
    ) -> Path:
        metrics_dir = self.subdir("metrics")
        path = metrics_dir / f"{name}.npz"

        # Drop None-valued metrics: np.asarray(None) yields an object array,
        # which np.savez cannot serialize (and which breaks the .ndim summary).
        metrics = {
            key: np.asarray(value)
            for key, value in metrics.items()
            if value is not None
        }

        np.savez(path, **metrics)

        if save_summary:
            summary = {
                key: float(value) for key, value in metrics.items() if value.ndim == 0
            }
            OmegaConf.save(
                OmegaConf.create(summary),
                metrics_dir / f"{name}_summary.yaml",
            )
        return path


def creat_exp_name(config: DictConfig) -> str:
    case = config.case.name
    da_method = config.da_method.name
    n_da = config.data_assimilation_steps
    dt_da = config.model_integration_steps
    M_da = config.ensemble_size
    base_name = f"{case}_{da_method}_M{M_da}_nsteps{n_da}_dtstep{dt_da}"

    # --- Add save
    name_appendix = config.save.savename_appendix
    if bool(name_appendix):
        name_appendix = name_appendix.strip()
        base_name += f"_{name_appendix}"
    return base_name


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
