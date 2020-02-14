find_package(HDF5 REQUIRED COMPONENTS C CXX)
target_include_directories(HOPS PUBLIC ${HDF5_INCLUDE_DIRS})
target_link_libraries(HOPS PUBLIC
        ${HDF5_LIBRARIES}
        stdc++fs
        )

target_sources(HOPS
        PUBLIC
        CsvWriter.hpp
        CsvWriterImpl.hpp
        FileWriter.hpp
        FileWriterFactory.hpp
        FileWriterType.hpp
        Hdf5Writer.hpp
        )

