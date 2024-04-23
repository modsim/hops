if (HOPS_MPI)
    find_package(MPI)

    if (MPI_FOUND AND UNIX)
        message(STATUS "Adding tests that require MPI")
        add_executable(ParallelTemperingTestSuite ParallelTemperingTestSuite.cpp)
        target_include_directories(ParallelTemperingTestSuite PRIVATE ${Boost_INCLUDE_DIRS} ${MPI_INCLUDE_PATH})
        target_link_libraries(ParallelTemperingTestSuite PRIVATE ${Boost_LIBRARIES})
        target_link_libraries(ParallelTemperingTestSuite PUBLIC hops)
        add_test(NAME ParallelTemperingTestSuite
                WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
                COMMAND ${MPIEXEC} --oversubscribe ${MPIEXEC_NUMPROC_FLAG} 3 ${CMAKE_CURRENT_BINARY_DIR}/ParallelTemperingTestSuite --log_format=JUNIT --log_sink=${PROJECT_BINARY_DIR}/test-reports/${TEST_NAME}.xml)
    endif (MPI_FOUND AND UNIX)
endif (HOPS_MPI)
