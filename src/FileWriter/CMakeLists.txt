find_package(HDF5 REQUIRED COMPONENTS C CXX)
target_include_directories(hops PUBLIC ${HDF5_INCLUDE_DIRS})
target_link_libraries(hops PUBLIC
        ${HDF5_LIBRARIES}
        stdc++fs
        )

target_sources(hops
        PRIVATE
        CsvWriter.cpp
        CsvWriterImpl.cpp
        FileWriterFactory.cpp
        Hdf5Writer.cpp
        )