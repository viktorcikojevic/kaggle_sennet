add_executable(solveWatershed
        solveWatershed.cpp
)
target_link_libraries(solveWatershed
        SennetLib
        CLI11::CLI11
        glog::glog
)

add_executable(mmapExample
        mmapExample.cpp
)
target_link_libraries(mmapExample
        SennetLib
)