set(TEST_SOURCES CsvWriterImplTest.cpp)

get_target_property(HOPS_HDF5_FOUND hops HOPS_HDF5_FOUND)
if (HOPS_HDF5_FOUND)
    message(STATUS "Adding Hdf5WriterTest")
    set(TEST_SOURCES ${TEST_SOURCES} Hdf5WriterTest.cpp)
endif (HOPS_HDF5_FOUND)


foreach (TEST_SOURCE ${TEST_SOURCES})
    get_filename_component(TEST_NAME ${TEST_SOURCE} NAME_WE)
    add_executable(${TEST_NAME} ${TEST_SOURCE})
    target_sources(runTests PRIVATE ${TEST_NAME})
    target_include_directories(${TEST_NAME} PUBLIC
            ../../include
            )
    target_link_libraries(${TEST_NAME}
            hops
            gtest
            gtest_main
            )
    gtest_discover_tests(${TEST_NAME})
endforeach (TEST_SOURCE)
