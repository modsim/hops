if (NOT HOPS_LIBRARY_TYPE STREQUAL "HEADER_ONLY")
    find_package(DNest4)

    if (DNest4_FOUND)
        message(STATUS "Building nested sampling tests because DNest4 was found")
    elseif (DNest4_FOUND)
        message(WARNING "Not building nested sampling tests because DNest4 was not found")
    endif (DNest4_FOUND)

endif (NOT HOPS_LIBRARY_TYPE STREQUAL "HEADER_ONLY")

set(TEST_SOURCES
        DNest4AdapterTestSuite.cpp
        )

foreach (TEST_SOURCE ${TEST_SOURCES})
    get_filename_component(TEST_NAME ${TEST_SOURCE} NAME_WE)
    add_executable(${TEST_NAME} ${TEST_SOURCE})
    target_link_libraries(${TEST_NAME} hops)
    add_test(NAME ${TEST_NAME}
            WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
            COMMAND ${TEST_NAME} --log_format=JUNIT --log_sink=${PROJECT_BINARY_DIR}/test-reports/${TEST_NAME}.xml --log_level=all --report_level=detailed)
endforeach (TEST_SOURCE)


# TODO TODO add postprocessing test to check evidence
