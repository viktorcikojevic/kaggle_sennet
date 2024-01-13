#include "WatershedSolver.h"
#include "fmt/format.h"
#include "glog/logging.h"
#include <deque>
#include <cmath>


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
    const std::vector<WatershedItem> neighbors {
            {point.x-1, point.y, point.z},
            {point.x+1, point.y, point.z},
            {point.x, point.y-1, point.z},
            {point.x, point.y+1, point.z},
            {point.x, point.y, point.z-1},
            {point.x, point.y, point.z+1},
    };
    return neighbors;
}


class WatershedSolver::Impl {
public:
    Impl(
            const std::shared_ptr<MmapArray>& image,
            const std::shared_ptr<MmapArray>& meanProb,
            const std::shared_ptr<MmapArray>& seed,
            double imageDiffThreshold,
            double labelUpperThreshold,
            double labelLowerBound
    ):
            image(image)
            , meanProb(meanProb)
            , seed(seed)
            , imageDiffThreshold(imageDiffThreshold)
            , labelUpperThreshold(labelUpperThreshold)
            , labelLowerBound(labelLowerBound)
    {
        checkSize(image, meanProb, "image", "meanProb");
        checkSize(image, seed, "image", "seed");
    }

    void solve() {
        std::deque<WatershedItem> deque;
        LOG(INFO) << "seeding";
        for (size_t z=0; z<seed->shape()[0]; ++z) {
            for (size_t y=0; y<seed->shape()[1]; ++y) {
                for (size_t x=0; x<seed->shape()[2]; ++x) {
                    deque.push_back({x, y, z});
                }
            }
        }
        LOG(INFO) << "seeded";

        while (!deque.empty()) {
            const auto point = deque[0];
            deque.pop_front();
            const auto neighbours = get6Neighbours(point);
            for (const auto& neighbour: neighbours) {
                checkNeighbour(deque, neighbour, point);
            }
        }
    }

    ~Impl() = default;
private:
    void checkNeighbour(std::deque<WatershedItem>& deque, const WatershedItem& neighbour, const WatershedItem& point) {
        if (!(
                -1 < neighbour.x && neighbour.x < image->shape()[2]
                && -1 < neighbour.y && neighbour.y < image->shape()[1]
                && -1 < neighbour.z && neighbour.z < image->shape()[0]
        )) {
            return;
        }
        if (bool(outputMask->get(neighbour.z, neighbour.y, neighbour.x))) {
            return;
        }
        const float newLabelVal = meanProb->get(neighbour.z, neighbour.y, neighbour.x);
        if (newLabelVal < labelLowerBound) {
            return;
        }
        const auto absImageDiff = std::abs(image->get(neighbour.z, neighbour.y, neighbour.x) - image->get(point.z, point.y, point.x));
        const auto imagePassesCheck = absImageDiff < imageDiffThreshold || newLabelVal > labelUpperThreshold;
        if (imagePassesCheck) {
            outputMask->set(neighbour.z, neighbour.y, neighbour.x, true);
            deque.push_back(neighbour);
        }
    }
    std::shared_ptr<MmapArray> image;
    std::shared_ptr<MmapArray> meanProb;
    std::shared_ptr<MmapArray> seed;
    std::shared_ptr<MmapArray> outputMask;
    double imageDiffThreshold;
    double labelUpperThreshold;
    double labelLowerBound;
};


WatershedSolver::WatershedSolver(
        const std::shared_ptr<MmapArray>& image,
        const std::shared_ptr<MmapArray>& meanProb,
        const std::shared_ptr<MmapArray>& seed,
        double imageDiffThreshold,
        double labelUpperThreshold,
        double labelLowerBound
):
    impl(std::make_unique<Impl>(
            image,
            meanProb,
            seed,
            imageDiffThreshold,
            labelUpperThreshold,
            labelLowerBound
    ))
{}

void WatershedSolver::solve() {
    impl->solve();
}

WatershedSolver::~WatershedSolver() = default;
