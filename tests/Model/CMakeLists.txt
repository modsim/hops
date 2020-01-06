  set(TEST_SOURCES
          MultivariateGaussianModelTest.cpp
          )

  foreach (TEST_SOURCE ${TEST_SOURCES})
      get_filename_component(TEST_NAME ${TEST_SOURCE} NAME_WE)
      add_executable(${TEST_NAME} ${TEST_SOURCE})
      target_include_directories(${TEST_NAME} PUBLIC
              ../../include
              )
      target_link_libraries(${TEST_NAME}
              nups
              gtest
              gtest_main
              )
      gtest_discover_tests(${TEST_NAME})
  endforeach (TEST_SOURCE)
