from typing import Optional, Dict
from line_profiler_pycharm import profile
import numpy as np

class Normalise:
    def __init__(
            self,
            mean: float = 0.5,
            std: float = 0.235,
            normalisation_percentile: Optional[int] = None,
    ):
        self.mean = mean
        self.std = std
        self.normalisation_percentile = normalisation_percentile

    @profile
    def transform(self, results: Dict) -> Optional[Dict]:
        if self.normalisation_percentile:
            img = results["img"]
            pct_lb = np.percentile(img.flatten(), self.normalisation_percentile)
            pct_ub = np.percentile(img.flatten(), 100 - self.normalisation_percentile)
            results["img"] = (results["img"] - pct_lb) / (pct_ub - pct_lb + 1e-6)
        else:
            results["img"] = results["img"] / 255.0
        results["img"] -= self.mean
        results["img"] /= self.std
        return results

    def __repr__(self):
        repr_str = (f"{self.__class__.__name__}("
                    f"mean={self.mean}, "
                    f"std={self.std}, "
                    f"n_pct={self.normalisation_percentile})")
        return repr_str
