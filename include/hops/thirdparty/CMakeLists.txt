set(HIGHFIVE_USE_EIGEN ON)
set(HIGHFIVE_USE_BOOST ON)
set(HIGHFIVE_UNIT_TESTS OFF)
set(HIGHFIVE_EXAMPLES OFF)

add_subdirectory(HighFive)
add_dependencies(hops HighFive)
target_link_libraries(hops PRIVATE HighFive)
