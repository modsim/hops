find_package(MPI REQUIRED)

target_sources(hops PRIVATE ColdnessAttribute.hpp ParallelTempering.hpp)
target_include_directories(hops PUBLIC ${MPI_INCLUDE_PATH})
target_link_libraries(hops PUBLIC ${MPI_LIBRARIES})
