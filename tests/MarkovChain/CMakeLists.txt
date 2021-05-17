add_subdirectory(ParallelTempering)
add_subdirectory(Proposal)
add_subdirectory(Recorder)

set(TEST_SOURCES
        AcceptanceRateTunerTestSuite.cpp
        IsGetColdnessAvailableTestSuite.cpp
        IsGetExchangeAttemptProbabilityAvailableTestSuite.cpp
        IsResetAcceptanceRateAvailableTestSuite.cpp
        IsGetStepSizeAvailableTestSuite.cpp
        IsSetColdnessAvailableTestSuite.cpp
        IsSetExchangeAttemptProbabilityAvailableTestSuite.cpp
        IsSetFisherWeightAvailableTestSuite.cpp
        IsSetStepSizeAvailableTestSuite.cpp
        MarkovChainFactoryTestSuite.cpp
        )

foreach (TEST_SOURCE ${TEST_SOURCES})
    get_filename_component(TEST_NAME ${TEST_SOURCE} NAME_WE)
    add_executable(${TEST_NAME} ${TEST_SOURCE})
    target_link_libraries(${TEST_NAME} PRIVATE hops)
    add_test(NAME ${TEST_NAME}
            WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
            COMMAND ${TEST_NAME} --log_format=XML --log_sink=${PROJECT_BINARY_DIR}/test-reports/${TEST_NAME}.xml --log_level=all --report_level=detailed)
endforeach (TEST_SOURCE)
