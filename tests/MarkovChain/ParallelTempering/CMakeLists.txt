set(TEST_SOURCES ColdnessAttributeTest.cpp)

foreach (TEST_SOURCE ${TEST_SOURCES})
    get_filename_component(TEST_NAME ${TEST_SOURCE} NAME_WE)
    add_executable(${TEST_NAME} ${TEST_SOURCE})
     target_sources(runTests PRIVATE ${TEST_NAME})
    target_include_directories(${TEST_NAME} PUBLIC ../../../include)
    target_link_libraries(${TEST_NAME} hops gtest gtest_main)
    gtest_discover_tests(${TEST_NAME})
endforeach (TEST_SOURCE)


find_package(MPI)

if(MPI_FOUND)
    message(STATUS "Adding tests that require MPI.")
    add_executable(ParallelTemperingTest ParallelTemperingTest.cpp)
    target_include_directories(ParallelTemperingTest PUBLIC ../../../include ${MPI_INCLUDE_PATH})
    target_link_libraries(ParallelTemperingTest hops gtest gtest_main ${MPI_LIBRARIES})
    add_test(NAME ParallelTemperingTest COMMAND ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} 3 ${CMAKE_CURRENT_BINARY_DIR}/ParallelTemperingTest)
endif(MPI_FOUND)
