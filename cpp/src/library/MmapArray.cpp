#include "MmapArray.h"
#include <fstream>
#include <filesystem>
#include <nlohmann/json.hpp>
#include <mio/mmap.hpp>


using json = nlohmann::json;


class MmapArray::Impl {
public:
    explicit Impl(const std::string& path):
        path(path),
        data(dataPath.string())
    {
        std::ifstream f(jsonPath);
        const json metaData = json::parse(f);
        metaData["shape"].get_to(mmapShape);
        metaData["dtype"].get_to(dtype);
        metaData["order"].get_to(order);

        if (order != "C") {
            throw std::runtime_error("don't know how to handle order " + order);
        }
        if (dtype == "uint8" || dtype == "bool") {
            stride = 1;
        } else if (dtype == "float") {
            stride = 8;  // TODO(Sumo): change to float when npy is fixed
        } else {
            throw std::runtime_error("don't know how to handle dtype " + dtype);
        }
        if (mmapShape.size() != 3) {
            throw std::runtime_error("can't handle shape.size() != 3 yet");
        }
    }

    ~Impl() = default;

    [[nodiscard]] std::vector<size_t> shape() const {
        return mmapShape;
    }

    [[nodiscard]] float get(size_t z, size_t y, size_t x) const {
        const size_t offset = coordToOffset(z, y, x);
        const auto* ptr = data.data() + offset;
        if (dtype == "uint8") {
            unsigned char ret;
            memcpy(&ret, ptr, stride);
            return float(ret);
        } else if (dtype == "float") {
            double ret;  // TODO(Sumo): change to float when npy is fixed
            memcpy(&ret, ptr, stride);
            return float(ret);
        } else if (dtype == "bool") {
            bool ret;
            memcpy(&ret, ptr, stride);
            return float(ret);
        } else {
            throw std::runtime_error("don't know how to handle dtype " + dtype);
        }
    }

    void set(size_t z, size_t y, size_t x, float val) {
        const size_t offset = coordToOffset(z, y, x);
        auto* ptr = data.data() + offset;
        if (dtype == "uint8") {
            auto charVal = (unsigned char)val;
            memcpy(ptr, &charVal, stride);
        } else if (dtype == "float") {
            double doubleVal = val;  // TODO(Sumo): change to float when npy is fixed
            memcpy(ptr, &doubleVal, stride);
        } else if (dtype == "bool") {
            auto charVal = (unsigned char)val;  // numpy actually writes this as a char
            memcpy(ptr, &charVal, stride);
        } else {
            throw std::runtime_error("don't know how to handle dtype " + dtype);
        }
    }

private:
    [[nodiscard]] size_t coordToOffset(size_t z, size_t y, size_t x) const {
        const size_t offset = (
                (z * (mmapShape[1] * mmapShape[2] * stride))
                + (y * (mmapShape[2] * stride))
                + (x * (stride))
        );
        return offset;
    }

    std::filesystem::path path;
    std::filesystem::path dataPath = path / "data.npy";
    std::filesystem::path jsonPath = path / "metadata.json";

    mio::mmap_sink data;

    std::vector<size_t> mmapShape;
    std::string dtype;
    std::string order;
    size_t stride;
};

MmapArray::MmapArray(const std::string &path):
    impl(std::make_unique<Impl>(path))
{}

MmapArray::~MmapArray() = default;

std::vector<size_t> MmapArray::shape() const {
    return impl->shape();
}

float MmapArray::get(size_t z, size_t y, size_t x) const {
    return impl->get(z, y, x);
}

void MmapArray::set(size_t z, size_t y, size_t x, float val) {
    impl->set(z, y, x, val);
}
