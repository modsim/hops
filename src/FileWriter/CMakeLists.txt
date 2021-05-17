find_package(HDF5 COMPONENTS C CXX)

if (HDF5_FOUND)
    message(STATUS "Found HDF5 libraries.")
    target_include_directories(hops PUBLIC ${HDF5_INCLUDE_DIRS})
    target_link_libraries(hops PUBLIC ${HDF5_LIBRARIES})
    target_compile_definitions(hops PRIVATE HOPS_HDF5_FOUND)

    target_sources(hops PRIVATE Hdf5Writer.cpp)
else ()
    message(STATUS "HDF5 libraries not found. Continuing without HDF5 support.")
endif ()

target_sources(hops
        PRIVATE
        CsvWriter.cpp
        CsvWriterImpl.cpp
        FileWriterFactory.cpp
        )

if ((CMAKE_CXX_COMPILER_ID STREQUAL "Clang") OR (CMAKE_CXX_COMPILER_ID STREQUAL "GNU"))
    target_link_libraries(hops PUBLIC stdc++fs)
elseif (CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
endif ()
