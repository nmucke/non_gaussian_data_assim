from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import jax.numpy as jnp
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
        arrays_dir = self.subdir("arrays")
        array_path = arrays_dir / f"{name}.npy"
        jnp.save(array_path, array, allow_pickle=False)
        return array_path


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


# def create_experiment_folder(save_name: str, root: Path) -> Path:
#     """Create unique fodler-name where to save yaml and expeiremnt"""
#     root.mkdir(parents=True, exist_ok=True)
#     base_path = root / save_name
#     idx = 0
#     while True:
#         exp_path = base_path if idx == 0 else root / f"{save_name}_{idx}"
#         try:
#             exp_path.mkdir()
#             return exp_path
#         except FileExistsError:
#             idx += 1

# def save_config(config: dict, save_path_exp: Path) -> None:
#     # --- Check if folder-nmae is still free, if so, create folder
#     save_path_yml = save_path_exp / "config"
#     save_path_yml.mkdir()
#     OmegaConf.save(config, save_path_yml / "config.yaml")


# def create_timing_name(base_path: Path, save_name: str) -> Path:
#     timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
#     save_path_exp = base_path / f"{save_name}_{timestamp}"
#     save_path_exp.mkdir(parents=True)
#     return save_path_exp
