if (HOPS_IO)
    set(TEST_SOURCES CsvReaderTestSuite.cpp)

    foreach (TEST_SOURCE ${TEST_SOURCES})
        get_filename_component(TEST_NAME ${TEST_SOURCE} NAME_WE)
        add_executable(${TEST_NAME} ${TEST_SOURCE})
        target_include_directories(${TEST_NAME} PRIVATE ${Boost_INCLUDE_DIRS})
        target_link_libraries(${TEST_NAME} ${Boost_LIBRARIES} hops)
        add_test(NAME ${TEST_NAME}
                WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
                COMMAND ${TEST_NAME} --log_format=JUNIT --log_sink=${PROJECT_BINARY_DIR}/tests/reports/${TEST_NAME}.xml)
    endforeach (TEST_SOURCE)
endif (HOPS_IO)
