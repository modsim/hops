if (HOPS_MPI_SUPPORTED)
    set(BIN_SOURCES ParallelTemperingDemo.cpp SamplingGaussianTargetDemo.cpp)
endif (HOPS_MPI_SUPPORTED)

foreach (BIN_SOURCE ${BIN_SOURCES})
    get_filename_component(BIN_NAME ${EXAMPLE_SOURCE} NAME_WE)
    add_executable(${BIN_NAME} ${EXAMPLE_SOURCE})
    target_link_libraries(${BIN_NAME} hops)
endforeach (BIN_SOURCE)

find_package(Boost COMPONENTS program_options REQUIRED)
add_executable(hops-sampler hops-sampler.cpp hops-sampler.hpp)
target_link_libraries(hops-sampler PRIVATE hops ${Boost_LIBRARIES})



