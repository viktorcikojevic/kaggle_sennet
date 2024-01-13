#pragma once


#include <vector>
#include <string>
#include <memory>


class MmapArray {
public:
    explicit MmapArray(const std::string& path);
    ~MmapArray();
    [[nodiscard]] std::vector<size_t> shape() const;
    [[nodiscard]] float get(size_t z, size_t y, size_t x) const;
    void set(size_t z, size_t y, size_t x, float val);

private:
    class Impl;
    std::unique_ptr<Impl> impl;
};
