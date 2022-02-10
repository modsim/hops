find_package(Boost COMPONENTS program_options REQUIRED)
add_executable(hops-sampler hops-sampler.cpp hops-sampler.hpp)
target_link_libraries(hops-sampler PRIVATE hops ${Boost_LIBRARIES})



