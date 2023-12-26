from line_profiler_pycharm import profile


class Normalise:
    def __init__(
            self,
            mean: float = 0.5,
            std: float = 0.235,
            normalise_by_mean_and_std: bool = False,
            normalisation_percentile: int | None = None,
    ):
        self.mean = mean
        self.std = std
        self.normalise_by_mean_and_std = normalise_by_mean_and_std
        self.normalisation_percentile = normalisation_percentile

    @profile
    def transform(self, results: dict) -> dict:
        if self.normalise_by_mean_and_std:
            results["img"] = (results["img"] - results["mean"]) / results["std"]
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
        return results

    def __repr__(self):
        repr_str = (f"{self.__class__.__name__}("
                    f"mean={self.mean}, "
                    f"std={self.std}, "
                    f"n_pct={self.normalisation_percentile})")
        return repr_str
