#include "WatershedSolver.h"
#include "fmt/format.h"
#include "glog/logging.h"
#include <deque>
#include <cmath>


//#pragma clang optimize off


void checkSize(
        const std::shared_ptr<MmapArray>& a,
        const std::shared_ptr<MmapArray>& b,
        const std::string& nameA,
        const std::string& nameB
) {
    if (a->shape().size() != b->shape().size()) {
        throw std::runtime_error(fmt::format("{}->shape() != {}->shape()", nameA, nameB));
    }
    for (size_t i=0; i<a->shape().size(); ++i) {
        if (a->shape()[i] != b->shape()[i]) {
            throw std::runtime_error(fmt::format("{}->shape() != {}->shape()", nameA, nameB));
        }
    }
}


struct WatershedItem {
    size_t x;
    size_t y;
    size_t z;
};


std::vector<WatershedItem> get6Neighbours(const WatershedItem& point) {
    std::vector<WatershedItem> neighbors {
            {point.x+1, point.y, point.z},
            {point.x, point.y+1, point.z},
            {point.x, point.y, point.z+1},
    };
    if (point.x > 0) {
        neighbors.push_back({point.x-1, point.y, point.z});
    }
    if (point.y > 0) {
        neighbors.push_back({point.x, point.y-1, point.z});
    }
    if (point.z > 0) {
        neighbors.push_back({point.x, point.y, point.z-1});
    }
    return neighbors;
}


class WatershedSolver::Impl {
public:
    Impl(
            const std::shared_ptr<MmapArray>& image,
            const std::shared_ptr<MmapArray>& meanProb,
            const std::shared_ptr<MmapArray>& seed,
            const std::shared_ptr<MmapArray>& outputMask,
            double imageDiffThreshold,
            double labelUpperThreshold,
            double labelLowerBound
    ):
            image(image)
            , meanProb(meanProb)
            , seed(seed)
            , outputMask(outputMask)
            , imageDiffThreshold(imageDiffThreshold)
            , labelUpperThreshold(labelUpperThreshold)
            , labelLowerBound(labelLowerBound)
    {
        checkSize(image, meanProb, "image", "meanProb");
        checkSize(image, seed, "image", "seed");
        checkSize(image, outputMask, "image", "outputMask");
    }

    void solve() {
        std::deque<WatershedItem> deque;
        initSeed(deque);
        const auto nStartingPoints = nFilledPoints;

        while (!deque.empty()) {
            LOG_EVERY_T(INFO, 5.0) << fmt::format("deque.size()={}, nFilledPoints={}", deque.size(), nFilledPoints);
//            LOG_EVERY_N(INFO, 1000) << fmt::format("deque.size()={}, nFilledPoints={}", deque.size(), nFilledPoints);
            const auto point = deque[0];
            deque.pop_front();
            const auto neighbours = get6Neighbours(point);
            for (const auto& neighbour: neighbours) {
                checkNeighbour(deque, neighbour, point);
            }
        }
        LOG(INFO) << "solver done: " << nStartingPoints << " -> " << nFilledPoints << " (" << nFilledPoints - nStartingPoints << ")";
    }

    ~Impl() = default;
private:
    void initSeed(std::deque<WatershedItem>& deque) {
        LOG(INFO) << "seeding";
        for (size_t z=0; z<seed->shape()[0]; ++z) {
            for (size_t y=0; y<seed->shape()[1]; ++y) {
                for (size_t x=0; x<seed->shape()[2]; ++x) {
                    if (bool(seed->get(z, y, x))) {
                        deque.push_back({x, y, z});
                        outputMask->set(z, y, x, true);
                        ++nFilledPoints;
//                        LOG(INFO) << fmt::format("seed: {}, {}, {}", z, y, x);
                    } else {
//                        outputMask->set(z, y, x, false);
                    }
                }
            }
            LOG_EVERY_N(INFO, 50) << "done slice: " << z << "/" << seed->shape()[0] << ": " << deque.size();
        }

//        deque.push_back({338, 928, 201});
//        outputMask->set(201, 928, 338, true);

        LOG(INFO) << "seeded";
    }
    void checkNeighbour(std::deque<WatershedItem>& deque, const WatershedItem& neighbour, const WatershedItem& point) {
        if (!(
                neighbour.x < image->shape()[2]
                && neighbour.y < image->shape()[1]
                && neighbour.z < image->shape()[0]
        )) {
//            LOG(INFO) << fmt::format("ob point: ({}, {}, {}) vs ({}, {}, {})", neighbour.z, neighbour.y, neighbour.x, image->shape()[0], image->shape()[1], image->shape()[2]);
            return;
        }
        const auto outputMaskVal = outputMask->get(neighbour.z, neighbour.y, neighbour.x);
        if (bool(outputMaskVal)) {
//            LOG(INFO) << fmt::format("mask true: ({}, {}, {})", neighbour.z, neighbour.y, neighbour.x);
            return;
        }
        const float newLabelVal = meanProb->get(neighbour.z, neighbour.y, neighbour.x);
        if (newLabelVal < labelLowerBound) {
//            LOG(INFO) << "low label: " << newLabelVal;
            return;
        }
        const auto absImageDiff = std::abs(image->get(neighbour.z, neighbour.y, neighbour.x) - image->get(point.z, point.y, point.x));
        const auto imagePassesCheck = absImageDiff < imageDiffThreshold || newLabelVal > labelUpperThreshold;
        if (imagePassesCheck) {
//            LOG(INFO) << "img pass check: " << absImageDiff << ", " << newLabelVal;
            outputMask->set(neighbour.z, neighbour.y, neighbour.x, true);
            deque.push_back(neighbour);
            ++nFilledPoints;
        } else {
//            LOG(INFO) << "img fail check: " << absImageDiff << ", " << newLabelVal;
        }
    }
    std::shared_ptr<MmapArray> image;
    std::shared_ptr<MmapArray> meanProb;
    std::shared_ptr<MmapArray> seed;
    std::shared_ptr<MmapArray> outputMask;
    double imageDiffThreshold;
    double labelUpperThreshold;
    double labelLowerBound;
    size_t nFilledPoints = 0;
};


WatershedSolver::WatershedSolver(
        const std::shared_ptr<MmapArray>& image,
        const std::shared_ptr<MmapArray>& meanProb,
        const std::shared_ptr<MmapArray>& seed,
        const std::shared_ptr<MmapArray>& outputMask,
        double imageDiffThreshold,
        double labelUpperThreshold,
        double labelLowerBound
):
    impl(std::make_unique<Impl>(
            image,
            meanProb,
            seed,
            outputMask,
            imageDiffThreshold,
            labelUpperThreshold,
            labelLowerBound
    ))
{}

void WatershedSolver::solve() {
    impl->solve();
}

WatershedSolver::~WatershedSolver() = default;
