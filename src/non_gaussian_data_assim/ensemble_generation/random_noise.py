import random

import numpy as np


class RandomNoise:
    """Class that handles the creation of random Noise (Red or White)"""

    def __init__(self, redness: float, seed: int | None = None) -> None:
        """
        Class that creates White or Red Noise.

        Parameter:
        ----------

        redness : int | None
            if redness=0 --> WhiteNoise

            default value for typical red-noise if alpha=2.5

            If int, random slope generation of spectrum: alpha is then between 2 and 3. If None is specified, I take a default value of 2.5
            Also, the seed will be used to specify the random seed. If it is None, a completely random (unreproducible) seed is chosen, otherwise I choose a seed 14*rednes

        alpha : float
            Power spectrum slope: P(k) ~ k^{-alpha}
        """
        # --- Inits
        if seed is None:
            seed = random.randrange(2**32)
            # seed = 14*int(redness)
        self.rng = np.random.default_rng(seed)

        # --- Redness (whiteness) of noise
        self.alpha = redness

    def generate_perturbation(
        self,
        shape_like: tuple,
        sigma: float,
    ) -> np.ndarray:
        """
        Generates a perturbation field. Either white Noise or spatially correlated.

        Args:
            shape_like: Size of perturbation field in following order (x,y,z)
            sigma (float): Scaling factor for noise. This is the desired standard-deviation of the field

        Returns:
            Perturbation field.
        """

        # ---------------------------------------------
        if len(shape_like) == 3:
            nx, ny, nz = shape_like[0], shape_like[1], shape_like[2]
            noise = self.create_rednoise(
                nx, ny, nz, normalize_amplitude=True
            )  # Perturbations are already scaled
        elif len(shape_like) == 2:
            nx, ny = shape_like[0], shape_like[1]
            noise = self.create_rednoise(
                nx, ny, normalize_amplitude=True
            )  # Perturbations are already scaled
        else:
            nx = shape_like[0]
            noise = self.create_rednoise(
                nx, normalize_amplitude=True
            )  # Perturbations are already scaled

        # --- Rescale field to unit variance ---
        noise_norm = noise / np.std(noise)
        # ---------------------------------------------

        # --- Scale standard deviation to desired level (sigma)
        perturbation = noise_norm * sigma  # Scale to desired standard-deviation
        return perturbation

    # =======================================================================================
    # -----------------------                RED NOISE                -----------------------
    # =======================================================================================

    def create_rednoise(
        self, *args: list, normalize_amplitude: bool = True
    ) -> np.ndarray:
        """
        Generate isotropic red noise with appropriate dimensionality.

        Inputs:
        *args : int
            Grid sizes. Length determines dimensionality:
            - 1 arg (nx,): 1D noise
            - 2 args (ny, nx): 2D noise
            - 3 args (nz, ny, nx): 3D noise

        Returns:
        field : ndarray
            Real-valued red-noise field with unit variance
        """
        ndim = len(args)
        if ndim < 1 or ndim > 3:
            raise ValueError(f"Expected 1-3 dimensions, got {ndim}")

        # --- 1) Build wavenumber arrays for each dimension
        k_arrays = [np.fft.fftfreq(n) for n in args]

        # --- 2) Create meshgrid with "ij" indexing for proper broadcasting
        if ndim == 1:
            k_grids = k_arrays
        else:
            k_grids = np.meshgrid(*k_arrays, indexing="ij")

        # --- 3) Set isotropic wavenumber magnitude
        k = np.sqrt(sum(kg**2 for kg in k_grids))
        k.flat[0] = np.inf  # note, this avoids divis by 0

        amplitude = k ** (-self.alpha / 2)

        # --- 4) Define Gaussian noise in Fourier space
        noise_hat = self.rng.normal(size=amplitude.shape) + 1j * self.rng.normal(
            size=amplitude.shape
        )

        # --- 5) Inverse FFT to physical space
        field = np.fft.irfftn(noise_hat * amplitude, s=args)

        if normalize_amplitude:
            field = field / field.std()

        return field
