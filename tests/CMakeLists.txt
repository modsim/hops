########################################################################################################################
# Test subdirectories
########################################################################################################################

find_package(Boost CONFIG REQUIRED COMPONENTS unit_test_framework)
set (Boost_USE_STATIC_LIBS OFF)

file(MAKE_DIRECTORY ${PROJECT_BINARY_DIR}/tests/reports)

add_subdirectory(FileReader)
add_subdirectory(FileWriter)
add_subdirectory(LinearProgram)
add_subdirectory(MarkovChain)
add_subdirectory(Model)
add_subdirectory(NestedSampling)
add_subdirectory(Optimization)
add_subdirectory(Polytope)
add_subdirectory(RandomNumberGenerator)
add_subdirectory(Statistics)
