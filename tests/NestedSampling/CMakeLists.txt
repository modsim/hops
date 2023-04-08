if (NOT HOPS_LIBRARY_TYPE STREQUAL "HEADER_ONLY")
	set(TEST_SOURCES
#			DiffusiveNestedSamplingTestSuite.cpp
#			LogLikelihoodValueTestSuite.cpp
			)
	if(HOPS_DNEST4_SUPPORT)
	   if (OpenMP_CXX_FOUND)
			set(TEST_SOURCES ${TEST_SOURCES} DNest4AdapterTestSuite.cpp)
		endif (OpenMP_CXX_FOUND)
	   configure_file(show_results.py ${CMAKE_CURRENT_BINARY_DIR} COPYONLY)
	endif(HOPS_DNEST4_SUPPORT)

	foreach (TEST_SOURCE ${TEST_SOURCES})
		get_filename_component(TEST_NAME ${TEST_SOURCE} NAME_WE)
		add_executable(${TEST_NAME} ${TEST_SOURCE})
		target_include_directories(${TEST_NAME} PRIVATE ${Boost_INCLUDE_DIRS})
		target_link_libraries (${TEST_NAME} ${Boost_LIBRARIES} hops)
		add_test(NAME ${TEST_NAME}
			WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
			COMMAND ${TEST_NAME} --log_format=JUNIT --log_sink=${PROJECT_BINARY_DIR}/test/reports/${TEST_NAME}.xml)
	endforeach (TEST_SOURCE)

endif (NOT HOPS_LIBRARY_TYPE STREQUAL "HEADER_ONLY")

