set(EXAMPLE_SOURCES
#        TODO
#        CoordinateHitAndRunExample.cpp
#        CoordinateHitAndRunRoundedExample.cpp
        SamplingGaussian.cpp
        )

foreach (EXAMPLE_SOURCE ${EXAMPLE_SOURCES})
    get_filename_component(EXAMPLE_NAME ${EXAMPLE_SOURCE} NAME_WE)
    add_executable(${EXAMPLE_NAME} ${EXAMPLE_SOURCE})
    target_link_libraries(${EXAMPLE_NAME} hops)
endforeach (EXAMPLE_SOURCE)

