set(TEST_SOURCES CsvReaderTest.cpp)

set(sbml_DIR /usr/local/lib/cmake CACHE PATH "Path to sbml-config.cmake. Usually found in /usr/local/cmake or /usr/local/lib/cmake on linux")

find_package(sbml CONFIG)

if(${SBML_FOUND})
    set(TEST_SOURCES ${TEST_SOURCES} SbmlReaderTest.cpp)
endif()


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
