set(EXAMPLE_SOURCES
        ParallelTemperingDemo.cpp
        SamplingUniformTarget.cpp
        SamplingGaussianTarget.cpp
        SamplingGaussianTargetDemo.cpp
        SamplingRBAModel.cpp
        SamplingRBAModelUniformly.cpp
        UniformSampler.cpp
        )

foreach (EXAMPLE_SOURCE ${EXAMPLE_SOURCES})
    get_filename_component(EXAMPLE_NAME ${EXAMPLE_SOURCE} NAME_WE)
    add_executable(${EXAMPLE_NAME} ${EXAMPLE_SOURCE})
    target_link_libraries(${EXAMPLE_NAME} hops)
endforeach (EXAMPLE_SOURCE)

