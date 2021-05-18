if (NOT HOPS_LIBRARY_TYPE STREQUAL "HEADER_ONLY")
    target_sources(hops
            PRIVATE
            CsvWriter.hpp
            CsvWriter.cpp
            CsvWriterImpl.cpp
            CsvWriterImpl.hpp
            FileWriter.hpp
            FileWriterFactory.hpp
            FileWriterFactory.cpp
            FileWriterType.hpp
            Hdf5Writer.hpp
            )
endif (NOT HOPS_LIBRARY_TYPE STREQUAL "HEADER_ONLY")

find_package(HDF5 COMPONENTS C CXX)
if (HDF5_FOUND AND HOPS_HDF5_SUPPORT)
    if (NOT HOPS_LIBRARY_TYPE STREQUAL "HEADER_ONLY")
        target_sources(hops ${SCOPE} Hdf5Writer.cpp)
    endif (NOT HOPS_LIBRARY_TYPE STREQUAL "HEADER_ONLY")
endif ()

if ((CMAKE_CXX_COMPILER_ID STREQUAL "Clang") OR (CMAKE_CXX_COMPILER_ID STREQUAL "GNU"))
    target_link_libraries(hops INTERFACE stdc++fs)
    target_link_libraries(hops ${SCOPE} stdc++fs)
elseif (CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
endif ()