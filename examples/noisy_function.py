# examples/noisy_function.py
from __future__ import annotations
import numpy as np
from typing import Tuple, Optional

class HeteroskedasticQuadratic:
    """
    Noisy heteroskedastic quadratic:
      true_f(x) = sum(x_i^2)
      y(x) = true_f(x) + Normal(0, sigma(x)^2)
      sigma(x) = sigma_min + (sigma_max - sigma_min) * tanh(alpha * ||x||)

    Use:
      prob = HeteroskedasticQuadratic(dim=5, seed=0)
      y, sigma = prob.noisy_f(np.array([1.0, 2.0, ...]))
    """

    def __init__(
        self,
        dim: int,
        sigma_min: float = 0.05,
        sigma_max: float = 0.5,
        alpha: float = 0.5,
        seed: Optional[int] = 0,
    ):
        assert dim >= 1
        assert 0.0 < sigma_min < sigma_max
        self.dim = dim
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.alpha = alpha
        self.rng = np.random.default_rng(seed)

    def true_f(self, x: np.ndarray) -> float:
        x = np.asarray(x, dtype=float)
        return float(np.sum(x**2))

    def sigma_of_x(self, x: np.ndarray) -> float:
        x = np.asarray(x, dtype=float)
        r = float(np.linalg.norm(x))
        t = np.tanh(self.alpha * r)  # in [0,1)
        return self.sigma_min + (self.sigma_max - self.sigma_min) * t

    def noisy_f(self, x: np.ndarray, deterministic: bool = False) -> Tuple[float, float]:
        """Return (y, sigma). If deterministic=True, returns (true_f(x), sigma)."""
        sig = self.sigma_of_x(x)
        if deterministic:
            return self.true_f(x), sig
        noise = self.rng.normal(0.0, sig)
        return self.true_f(x) + noise, sig


if __name__ == "__main__":
    # quick visual check of sigma profile
    import matplotlib.pyplot as plt
    dim = 2
    prob = HeteroskedasticQuadratic(dim=dim, seed=123)
    radii = np.linspace(0, 6, 200)
    sigmas = []
    for r in radii:
        x = np.array([r] + [0]*(dim-1), dtype=float)
        sigmas.append(prob.sigma_of_x(x))
    plt.figure()
    plt.plot(radii, sigmas)
    plt.xlabel("radius ||x||")
    plt.ylabel("sigma(x)")
    plt.title("Heteroskedastic noise profile")
    plt.tight_layout()
    plt.show()