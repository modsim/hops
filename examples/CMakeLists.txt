set(EXAMPLE_SOURCES
        SamplingGaussianTargetDemo.cpp
        SamplingGaussianSimplexDemo.cpp
        SamplingRosenbrockTargetDemo.cpp
        SamplingUniformTargetDemo.cpp
        )

if(HOPS_MPI_SUPPORTED)
    set(EXAMPLE_SOURCES ${EXAMPLE_SOURCES} ParallelTemperingDemo.cpp)
endif(HOPS_MPI_SUPPORTED)

foreach (EXAMPLE_SOURCE ${EXAMPLE_SOURCES})
    get_filename_component(EXAMPLE_NAME ${EXAMPLE_SOURCE} NAME_WE)
    add_executable(${EXAMPLE_NAME} ${EXAMPLE_SOURCE})
    target_link_libraries(${EXAMPLE_NAME} hops)
    target_include_directories(${EXAMPLE_NAME} PUBLIC  ../include)
endforeach (EXAMPLE_SOURCE)

