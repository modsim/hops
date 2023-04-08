set(TEST_SOURCES
        BilliardMALATestSuite.cpp
        ChordStepDistributionsTestSuite.cpp
        CSmMALATestSuite.cpp
        CoordinateHitAndRunTestSuite.cpp
        DikinEllipsoidCalculatorTestSuite.cpp
        DikinTestSuite.cpp
        HitAndRunTestSuite.cpp
        IsSetStepSizeAvailableTestSuite.cpp
        ReflectorTestSuite.cpp
        TrunatedGaussianProposalTestSuite.cpp
        )

if (CMAKE_BUILD_TYPE EQUAL "RELEASE")
    # Only run in release, because it is a slow test. Numerically there should be no difference for this test-case.
    set(TEST_SOURCES ${TEST_SOURCES} ReversibleJumpProposalTestSuite.cpp)
endif (CMAKE_BUILD_TYPE EQUAL "RELEASE")

foreach (TEST_SOURCE ${TEST_SOURCES})
    get_filename_component(TEST_NAME ${TEST_SOURCE} NAME_WE)
    add_executable(${TEST_NAME} ${TEST_SOURCE})
    target_include_directories(${TEST_NAME} PRIVATE ${Boost_INCLUDE_DIRS})
    target_link_libraries (${TEST_NAME} ${Boost_LIBRARIES} hops)
    add_test(NAME ${TEST_NAME}
            WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
            COMMAND ${TEST_NAME} --log_format=JUNIT --log_sink=${PROJECT_BINARY_DIR}/tests/reports/${TEST_NAME}.xml)
endforeach (TEST_SOURCE)

