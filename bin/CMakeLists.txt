find_package(MPI)

if (MPI_FOUND)
#    set(BIN_SOURCES ParallelTemperingDemo.cpp)
    set(BIN_SOURCES )
endif (MPI_FOUND)

foreach (BIN_SOURCE ${BIN_SOURCES})
    get_filename_component(BIN_NAME ${BIN_SOURCE} NAME_WE)
    add_executable(${BIN_NAME} ${BIN_SOURCE})
    target_link_libraries(${BIN_NAME} hops)
endforeach (BIN_SOURCE)

find_package(Boost COMPONENTS program_options REQUIRED)
add_executable(hops-sampler hops-sampler.cpp hops-sampler.hpp)
target_link_libraries(hops-sampler PRIVATE hops ${Boost_LIBRARIES})

add_executable(sample-linear-model sample-linear-model.cpp hops-sampler.hpp)
target_link_libraries(sample-linear-model PRIVATE hops)



