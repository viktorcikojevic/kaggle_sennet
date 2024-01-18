#include <string>
#include <iostream>
#include "MmapArray.h"


int main() {
    {
        const std::string path = "/home/clay/research/kaggle/sennet/data_dumps/processed/kidney_1_dense/image";
        const auto mmap = MmapArray(path);
        const auto shape = mmap.shape();
        std::cout << "shape[0] = " << shape[0] << "\n";
        std::cout << "shape[1] = " << shape[1] << "\n";
        std::cout << "shape[2] = " << shape[2] << "\n";
        std::cout << "mmap[100, 200, 300] = " << mmap.get(100, 200, 300) << "\n";
    }
    {
        const std::string path = "/home/clay/research/kaggle/sennet/data_dumps/predicted/ensembled/kidney_3_sparse/chunk_00/mean_prob";
        const auto mmap = MmapArray(path);
        const auto shape = mmap.shape();
        std::cout << "shape[0] = " << shape[0] << "\n";
        std::cout << "shape[1] = " << shape[1] << "\n";
        std::cout << "shape[2] = " << shape[2] << "\n";
        std::cout << "mmap[100, 200, 300] = " << mmap.get(100, 200, 300) << "\n";
    }
}
