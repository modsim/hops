find_package(Celero REQUIRED)

set(BENCHMARK_SOURCES
        CsvReaderBenchmark.cpp
        )

foreach (BENCHMARK_SOURCE ${BENCHMARK_SOURCES})
    get_filename_component(BENCHMARK_NAME ${BENCHMARK_SOURCE} NAME_WE)
    add_executable(${BENCHMARK_NAME} ${BENCHMARK_SOURCE} Fixtures.hpp)
    target_include_directories(${BENCHMARK_NAME} PUBLIC
            ${EIGEN3_INCLUDE_DIR}
            ${CELERO_INCLUDE_DIRS}
            )
    target_link_libraries(${BENCHMARK_NAME}
            ${CELERO_LIBRARIES}
            hops
            )
endforeach (BENCHMARK_SOURCE)
