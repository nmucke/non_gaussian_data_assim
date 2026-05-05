"""Helpers for unified DA experiments: initial states and prior ensembles."""

from abc import ABC, abstractmethod
from typing import Literal, Optional

import jax
import jax.numpy as jnp

from non_gaussian_data_assim.ensemble_generation.random_noise import (
    RandomNoiseGenerator,
)

ensgen_method = Literal["white_noise", "red_noise", "breeding", "kuramoto_cosine"]


class BaseNoise(ABC):
    """
    Base Class that handles creation of Ensemble-perturbations (noise)
    """

    def __init__(
        self,
        name: str,
        rng_key: jax.Array,
        shape: tuple,
    ) -> None:
        self.name = name
        self.rng_key = rng_key
        self.shape = shape

    @abstractmethod
    def sample(self) -> jnp.ndarray:
        """Sample Ensemble-Perturbations for a given ensemble member"""
        raise NotImplementedError


class PriorEnsemble(ABC):
    """This class creates the prior ensemle by using the correct BaseNoise child"""

    def __init__(
        self,
        rng_key: jax.Array,
        ensemble_size: int,
        periodic: bool,
        shape_member: tuple,
        X_bestguess: Optional[jax.ndarray] = None,
    ) -> None:

        self.rng_key = rng_key
        self.ensemble_size = ensemble_size
        self.periodic = periodic
        self.shape_member = shape_member
        self.X_bestguess = X_bestguess

    def set_periodic_bc(self, X: jnp.ndarray) -> jnp.ndarray:
        """Enforce periodic boudaries"""
        X = X.at[..., -1].set(
            X[..., 0]
        )  # Assume that last dimension is spatial dimension
        return X

    @abstractmethod
    def sample(self) -> jnp.ndarray:
        """Sample an entire ensemble with the chosen ensemble-generation method"""
        raise NotImplementedError


# ===========================================================================================================================

# ----> This is just given by a single instance of Noise with the coreect magnitude / scale!

# class InitialConditions(ABC):
#     """
#     Create I.C. with specified shape and a magnitude (or standard deviation for random noise)
#     """
#     def __init__(
#         self,
#         shape:tuple,
#         periodic:bool,
#         magnitude:float
#         ) -> None:
#         self.shape = shape
#         self.periodic = periodic

#     @abstractmethod
#     def sample():
#         raise NotImplementedError
# ----------------> This is just an instance of BASE NOISE


class RandomNoise(BaseNoise):
    """Create White- or Red-Noise."""

    def __init__(
        self,
        rng_key: jax.Array,
        shape: tuple,
        scale: float,
        alpha: Optional[float] = None,
    ) -> None:

        name = "white_noise" if alpha is None or alpha == 0 else "red_noise"

        super().__init__(name=name, rng_key=rng_key, shape=shape)
        self.scale = scale
        self.alpha = alpha
        # --- Automaticcally chooses which method is used based on the value of alpha
        self._sampler = (
            self.white_noise if alpha is None or alpha == 0 else self.red_noise
        )

    @staticmethod
    def _seed_from_jax_key(rng_key: jax.Array) -> int:
        """Convert a jax key (rnpg) key to a numpy seed of type int."""
        seed = seed = jax.random.randint(rng_key, (), 0, 2**31 - 1)
        return int(jax.device_get(seed))

    def white_noise(self) -> jnp.ndarray:
        """Sample a white (in space) noise of shape [1, num_states, state_dim]."""
        # Same as X_0 = jax.random.normal(rng_key, (1, num_states, state_dim)) * scale. NOTE: Sshape must be (num_states, state_dim)
        x_dash = jax.random.normal(self.rng_key, (1, *self.shape)) * self.scale
        return x_dash

    def red_noise(self) -> jnp.ndarray:
        """Sample a red (in space) noise of shape [1, num_states, state_dim]."""

        seed = self._seed_from_jax_key(self.rng_key)
        noise_generator = RandomNoiseGenerator(redness=self.alpha, seed=seed)
        x_dash = noise_generator.generate_perturbation(
            shape_like=self.shape, sigma=self.scale  # (num_states, state_dim)
        )
        return x_dash

    def sample(self) -> jnp.ndarray:
        return self._sampler()


class CosineNoise(BaseNoise):
    """Deterministic two-mode cosine initial state for Kuramoto-Sivashinsky."""

    def __init__(
        self,
        rng_key: jax.Array,
        shape: tuple,  # is here  state_dim
        domain_length: float,
        magnitude: float,
    ) -> None:

        super().__init__(name="2mode_cosine_noise", rng_key=rng_key, shape=shape)
        self.domain_length = domain_length
        self.magnitude = magnitude

    def sample(self) -> jnp.ndarray:
        x = jnp.linspace(0.0, self.domain_length, self.shape)
        profile = self.magnitude * jnp.cos(
            2 * jnp.pi * x / self.domain_length
        ) + self.magnitude * jnp.cos(4 * jnp.pi * x / self.domain_length)
        return profile.reshape(1, 1, self.shape)


# def kuramoto_cosine_initial_state(
#     rng_key: jax.Array,
#     state_dim: int,
#     domain_length: float,
#     magnitude: float,
# ) -> jnp.ndarray:
#     """Deterministic two-mode cosine initial state for Kuramoto-Sivashinsky."""
#     del rng_key
#     x = jnp.linspace(0.0, domain_length, state_dim)
#     profile = magnitude * jnp.cos(2 * jnp.pi * x / domain_length) + magnitude * jnp.cos(
#         4 * jnp.pi * x / domain_length
#     )
#     return profile.reshape(1, 1, state_dim)


class RandomEnsemble(PriorEnsemble):

    def __init__(
        self,
        rng_key: jax.Array,
        ensemble_size: int,
        alpha: float,
        scale: float,
        periodic: bool,
        shape_member: tuple,  # This is given by   state_dim
        X_bestguess: Optional[jax.ndarray] = None,
    ) -> None:

        super().__init__(
            rng_key=rng_key,
            ensemble_size=ensemble_size,
            periodic=periodic,
            shape_member=shape_member,
            X_bestguess=X_bestguess,
        )
        self.scale = scale
        self.alpha = alpha

    def sample(self) -> jnp.ndarray:
        seeds_member = jax.random.split(self.rng_key, self.ensemble_size)
        ensemble_members = []
        for seed in seeds_member:
            noise_generator = RandomNoiseGenerator(redness=self.alpha, seed=seed)
            X_member = noise_generator.generate_perturbation(
                shape_like=self.shape_member, sigma=self.scale
            )

            if self.X_bestguess is not None:
                X_member += self.X_bestguess
            X_member = self.set_periodic_bc(X_member) if self.periodic else X_member

            ensemble_members.append(X_member)

        ensemble = jnp.stack(ensemble_members, axis=0)
        return ensemble


# OLD WAY:
# def white_noise_ensemble(
#     rng_key: jax.Array,
#     ensemble_size: int,
#     num_states: int,
#     state_dim: int,
#     scale: float,
#     periodic_boundary: bool,
# ) -> jnp.ndarray:
#     """Sample a random-normal prior ensemble of shape [ensemble_size, num_states, state_dim]."""
#     ensemble = (
#         jax.random.normal(rng_key, (ensemble_size, num_states, state_dim)) * scale
#     )

#     if periodic_boundary:
#         ensemble = ensemble.at[:, :, -1].set(ensemble[:, :, 0])
#     return ensemble


class CosineEnsemble(PriorEnsemble):
    def __init__(
        self,
        rng_key: jax.Array,
        ensemble_size: int,
        domain_length: float,
        magnitude_min: float,
        magnitude_max: float,
        periodic: bool,
        shape_member: tuple,  # This is given by   (state_dim,)
        X_bestguess: Optional[jax.ndarray] = None,
    ) -> None:

        super().__init__(
            rng_key=rng_key,
            ensemble_size=ensemble_size,
            periodic=periodic,
            shape_member=shape_member,
            X_bestguess=X_bestguess,
        )
        self.domain_length = domain_length
        self.magnitude_min = magnitude_min
        self.magnitude_max = magnitude_max

    def sample(self) -> jnp.ndarray:
        x = jnp.linspace(0.0, self.domain_length, self.shape_member)
        magnitudes = jax.random.uniform(
            self.rng_key,
            (self.ensemble_size,),
            minval=self.magnitude_min,
            maxval=self.magnitude_max,
        )

        def profile(m: jnp.ndarray) -> jnp.ndarray:
            return m * jnp.cos(2 * jnp.pi * x / self.domain_length) + m * jnp.cos(
                4 * jnp.pi * x / self.domain_length
            )

        profiles = jax.vmap(profile)(magnitudes)
        return profiles.reshape(self.ensemble_size, 1, self.shape_member)


# OLD WAY:
# def kuramoto_cosine_prior_ensemble(
#     rng_key: jax.Array,
#     ensemble_size: int,
#     state_dim: int,
#     domain_length: float,
#     magnitude_min: float,
#     magnitude_max: float,
# ) -> jnp.ndarray:
#     """Prior ensemble for Kuramoto-Sivashinsky with magnitudes drawn uniformly."""

#     x = jnp.linspace(0.0, domain_length, state_dim)
#     magnitudes = jax.random.uniform(
#         rng_key, (ensemble_size,), minval=magnitude_min, maxval=magnitude_max
#     )

#     def profile(m: jnp.ndarray) -> jnp.ndarray:
#         return m * jnp.cos(2 * jnp.pi * x / domain_length) + m * jnp.cos(
#             4 * jnp.pi * x / domain_length
#         )

#     profiles = jax.vmap(profile)(magnitudes)
#     return profiles.reshape(ensemble_size, 1, state_dim)
