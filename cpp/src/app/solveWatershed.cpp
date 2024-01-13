#include "MmapArray.h"
#include "WatershedSolver.h"
#include "CLI/CLI.hpp"
#include "glog/logging.h"
#include <filesystem>


int main(int argc, char **argv) {
    std::filesystem::path imageMmapPath;
    std::filesystem::path meanPredMmapPath;
    std::filesystem::path seedMmapPath;
    std::filesystem::path outMmapPath;
    double imageDiffThreshold;
    double labelUpperBound;
    double labelLowerBound;

    CLI::App app{"solveWatershed"};
    app.add_option("--image", imageMmapPath, "imageMmapPath")
            ->required()
            ->check(CLI::ExistingDirectory);
    app.add_option("--pred", meanPredMmapPath, "meanPredMmapPath")
            ->required()
            ->check(CLI::ExistingDirectory);
    app.add_option("--seed", seedMmapPath, "seedMmapPath")
            ->required()
            ->check(CLI::ExistingDirectory);
    app.add_option("--out", outMmapPath, "outMmapPath")
            ->required()
            ->check(CLI::ExistingDirectory);
    app.add_option("--image-diff-threshold", imageDiffThreshold, "imageDiffThreshold")->required();
    app.add_option("--label-upper-bound", labelUpperBound, "labelUpperBound")->required();
    app.add_option("--label-lower-bound", labelLowerBound, "labelLowerBound")->required();
    CLI11_PARSE(app, argc, argv);

    auto image = std::make_shared<MmapArray>(imageMmapPath);
    auto pred = std::make_shared<MmapArray>(meanPredMmapPath);
    auto seed = std::make_shared<MmapArray>(seedMmapPath);
    auto out = std::make_shared<MmapArray>(outMmapPath);
    WatershedSolver solver{
        image,
        pred,
        seed,
        out,
        imageDiffThreshold,
        labelUpperBound,
        labelLowerBound,
    };
    solver.solve();
    return 0;
}
