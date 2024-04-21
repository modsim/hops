# Avoid having to make Header only version of deprecated MaximumVolumeEllipsoid by only compiling these
# Examples when hops is not in HEADER_ONLY mode.
if (NOT HOPS_LIBRARY_TYPE STREQUAL "HEADER_ONLY")
    set(BIN_SOURCES
            SamplingGaussianTarget.cpp
            SamplingGaussianSignificanceTarget.cpp
            TruncatedGaussianDemo.cpp
    )

    if (HOPS_MPI)
        set(BIN_SOURCES ${BIN_SOURCES} GaussianDemo.cpp)
    endif (HOPS_MPI)


    foreach (BIN_SOURCE ${BIN_SOURCES})
        get_filename_component(BIN_NAME ${BIN_SOURCE} NAME_WE)
        add_executable(${BIN_NAME} ${BIN_SOURCE})
        target_link_libraries(${BIN_NAME} hops ${TBB_LIBRARIES})
    endforeach (BIN_SOURCE)


endif (NOT HOPS_LIBRARY_TYPE STREQUAL "HEADER_ONLY")
