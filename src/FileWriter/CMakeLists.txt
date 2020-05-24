find_package(HDF5 COMPONENTS C CXX)

if (${HDF_FOUND})
    # TODO add definition hdf5 found
    target_include_directories(hops PUBLIC ${HDF5_INCLUDE_DIRS})
    target_link_libraries(hops PUBLIC ${HDF5_LIBRARIES})

    target_sources(hops PRIVATE Hdf5Writer.cpp)
endif ()

target_sources(hops
        PRIVATE
        CsvWriter.cpp
        CsvWriterImpl.cpp
        FileWriterFactory.cpp
        )

if ((CMAKE_CXX_COMPILER_ID STREQUAL "Clang") OR (CMAKE_CXX_COMPILER_ID STREQUAL "GNU"))
    target_link_libraries(hops PUBLIC
            stdc++fs
            )
elseif (CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
endif ()


