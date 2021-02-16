if (HOPS_MPI_SUPPORTED)
    set(BIN_SOURCES ${EXAMPLE_SOURCES} ParallelTemperingDemo.cpp)
endif (HOPS_MPI_SUPPORTED)

foreach (BIN_SOURCE ${EXAMPLE_SOURCES})
    get_filename_component(BIN_NAME ${EXAMPLE_SOURCE} NAME_WE)
    add_executable(${BIN_NAME} ${EXAMPLE_SOURCE})
    target_link_libraries(${BIN_NAME} hops)
    target_include_directories(${BIN_NAME} PUBLIC ../include)
endforeach (BIN_SOURCE)

find_package(Boost COMPONENTS program_options REQUIRED)
add_executable(hops-sampler hops-sampler.cpp hops-sampler.hpp)
target_link_libraries(hops-sampler hops ${Boost_LIBRARIES})
target_include_directories(hops-sampler PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../include/${LIBRARY_NAME}>
        $<INSTALL_INTERFACE:../include/${LIBRARY_NAME}>
        )



