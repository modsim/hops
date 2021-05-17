target_sources(hops PRIVATE hops.hpp)

add_subdirectory(Diagnostics)
add_subdirectory(FileReader)
add_subdirectory(FileWriter)
add_subdirectory(LinearProgram)
add_subdirectory(MarkovChain)
add_subdirectory(Model)
add_subdirectory(Polytope)
add_subdirectory(RandomNumberGenerator)
add_subdirectory(third-party)
add_subdirectory(Transformation)
add_subdirectory(Utility)

