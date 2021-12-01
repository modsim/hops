set(TEST_SOURCES ColdnessTestSuite.cpp)

foreach (TEST_SOURCE ${TEST_SOURCES})
    get_filename_component(TEST_NAME ${TEST_SOURCE} NAME_WE)
    add_executable(${TEST_NAME} ${TEST_SOURCE})
    target_link_libraries(${TEST_NAME} hops)
    add_test(NAME ${TEST_NAME}
            WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
            COMMAND ${TEST_NAME} --log_format=JUNIT --log_sink=${PROJECT_BINARY_DIR}/tests/reports/${TEST_NAME}.xml --log_level=all --report_level=detailed)
endforeach (TEST_SOURCE)


# TODO reinclude
#find_package(MPI)
#
#if(MPI_FOUND AND UNIX)
#    message(STATUS "Adding tests that require MPI")
#    add_executable(ParallelTemperingTestSuite ParallelTemperingTestSuite.cpp)
#    target_include_directories(ParallelTemperingTestSuite PRIVATE ${MPI_INCLUDE_PATH})
#    target_link_libraries(ParallelTemperingTestSuite ${MPI_LIBRARIES} hops)
#    add_test(NAME ParallelTemperingTestSuite
#            WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
#            COMMAND ${MPIEXEC} --oversubscribe ${MPIEXEC_NUMPROC_FLAG} 3 ${CMAKE_CURRENT_BINARY_DIR}/ParallelTemperingTestSuite --log_format=JUNIT --log_sink=${PROJECT_BINARY_DIR}/test-reports/${TEST_NAME}.xml --log_level=all --report_level=detailed)
#endif(MPI_FOUND AND UNIX)
