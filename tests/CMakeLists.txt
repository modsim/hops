########################################################################################################################
# Test subdirectories
########################################################################################################################

project(hops-tests CXX)

find_package(Boost REQUIRED COMPONENTS unit_test_framework)

file(MAKE_DIRECTORY ${PROJECT_BINARY_DIR}/test-reports)

add_subdirectory(FileReader)
add_subdirectory(FileWriter)
add_subdirectory(LinearProgram)
add_subdirectory(MarkovChain)
add_subdirectory(Model)
add_subdirectory(PolytopePreprocessing)
