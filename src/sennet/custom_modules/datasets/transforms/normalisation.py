from line_profiler_pycharm import profile
import numpy as np


class Normalise:
    def __init__(
            self,
            mean: float = 0.5,
            std: float = 0.235,
            normalise_by_mean_and_std: bool = False,
            normalisation_percentile: int | None = None,
            leak_gradient: float | None = None,
    ):
        self.mean = mean
        self.std = std
        self.normalise_by_mean_and_std = normalise_by_mean_and_std
        self.normalisation_percentile = normalisation_percentile
        self.leak_gradient = leak_gradient

    @profile
    def transform(self, results: dict) -> dict:
        if self.normalise_by_mean_and_std:
            results["img"] = (results["img"] - results["mean"]) / results["std"]
            if self.leak_gradient is not None:
                high_z = 5
                low_z = -3
                high_z_mask = results["img"] > high_z
                low_z_mask = results["img"] < low_z
                results["img"][high_z_mask] = (results["img"][high_z_mask] - high_z) * self.leak_gradient + high_z
                results["img"][low_z_mask] = (results["img"][low_z_mask] - low_z) * self.leak_gradient + low_z
        elif self.normalisation_percentile:
            pct_lb = results[f"percentile_{self.normalisation_percentile}"]
            pct_ub = results[f"percentile_{100 - self.normalisation_percentile}"]
            results["img"] = (results["img"] - pct_lb) / (pct_ub - pct_lb + 1e-6)
            results["img"] -= self.mean
            results["img"] /= self.std
        else:
            results["img"] = results["img"] / 255.0
            results["img"] -= self.mean
            results["img"] /= self.std
        results["img"] = results["img"].astype(np.float32)  # makes downstream albumentations happy
        return results

    def __repr__(self):
        repr_str = (f"{self.__class__.__name__}("
                    f"mean={self.mean}, "
                    f"std={self.std}, "
                    f"n_pct={self.normalisation_percentile})")
        return repr_str
