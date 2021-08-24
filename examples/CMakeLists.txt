set(BIN_SOURCES
        SamplingGaussianTarget.cpp
        SimplexBenchmarks.cpp
        )

find_package(MPI)
if (MPI_FOUND)
    set(BIN_SOURCES ${BIN_SOURCES} ParallelTempering.cpp)
endif (MPI_FOUND)

find_package(TBB REQUIRED)

foreach (BIN_SOURCE ${BIN_SOURCES})
    get_filename_component(BIN_NAME ${BIN_SOURCE} NAME_WE)
    add_executable(${BIN_NAME} ${BIN_SOURCE})
    target_link_libraries(${BIN_NAME} hops ${TBB_LIBRARIES})
endforeach (BIN_SOURCE)


