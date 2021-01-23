set(TEST_SOURCES CsvWriterImplTestSuite.cpp)

get_target_property(HOPS_HDF5_FOUND hops HOPS_HDF5_FOUND)
if (HOPS_HDF5_FOUND)
    message(STATUS "Adding Hdf5WriterTest")
    set(TEST_SOURCES ${TEST_SOURCES} Hdf5WriterTestSuite.cpp)
endif (HOPS_HDF5_FOUND)


foreach (TEST_SOURCE ${TEST_SOURCES})
    get_filename_component(TEST_NAME ${TEST_SOURCE} NAME_WE)
    add_executable(${TEST_NAME} ${TEST_SOURCE})
    target_include_directories(${TEST_NAME} PUBLIC
            $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include/>
            $<INSTALL_INTERFACE:include/${LIBRARY_NAME}>
            )
    target_link_libraries(${TEST_NAME} hops)
    add_test(NAME ${TEST_NAME}
            WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
            COMMAND ${TEST_NAME} --log_format=XML --log_sink=${PROJECT_BINARY_DIR}/test-reports/${TEST_NAME}.xml --log_level=all --report_level=detailed)
endforeach (TEST_SOURCE)
