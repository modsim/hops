if (NOT HOPS_LIBRARY_TYPE STREQUAL "HEADER_ONLY")
    target_sources(hops PRIVATE hops.hpp)
endif (NOT HOPS_LIBRARY_TYPE STREQUAL "HEADER_ONLY")

add_subdirectory(Statistics)
add_subdirectory(FileReader)
add_subdirectory(FileWriter)
add_subdirectory(LinearProgram)
add_subdirectory(MarkovChain)
add_subdirectory(Model)
add_subdirectory(Polytope)
add_subdirectory(RandomNumberGenerator)
add_subdirectory(Third-party)
add_subdirectory(Transformation)
add_subdirectory(Utility)

