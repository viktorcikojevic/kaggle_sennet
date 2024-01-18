#pragma once

#include <string>
#include <memory>
#include <MmapArray.h>


class WatershedSolver {
public:
    WatershedSolver(
            const std::shared_ptr<MmapArray>& image,
            const std::shared_ptr<MmapArray>& meanProb,
            const std::shared_ptr<MmapArray>& seed,
            const std::shared_ptr<MmapArray>& outputMask,
            double imageDiffThreshold,
            double labelUpperThreshold,
            double labelLowerBound
    );
    void solve();
    ~WatershedSolver();
private:
    class Impl;
    std::unique_ptr<Impl> impl;
};
