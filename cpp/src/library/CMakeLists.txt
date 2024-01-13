add_library(SennetLib STATIC
        MmapArray.cpp MmapArray.h
        WatershedSolver.cpp WatershedSolver.h
)
target_link_libraries(SennetLib PRIVATE
        mio::mio
        nlohmann_json::nlohmann_json
        fmt::fmt
        glog::glog
)