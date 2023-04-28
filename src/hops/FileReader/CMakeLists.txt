if (NOT HOPS_LIBRARY_TYPE STREQUAL "HEADER_ONLY")
    target_sources(hops PRIVATE
            CsvReader.hpp
            CsvReader.cpp
            Hdf5Reader.hpp
            SbmlModel.hpp
            SbmlReader.hpp
            )
endif (NOT HOPS_LIBRARY_TYPE STREQUAL "HEADER_ONLY")

find_package(SBML)
if (SBML_FOUND AND HOPS_SBML_SUPPORT)
    message(STATUS "SBML found.")
    target_link_libraries(hops INTERFACE sbml)
    target_link_libraries(hops ${SCOPE} sbml)
    if (NOT HOPS_LIBRARY_TYPE STREQUAL "HEADER_ONLY")
        target_sources(hops PRIVATE SbmlReader.cpp)
    endif (NOT HOPS_LIBRARY_TYPE STREQUAL "HEADER_ONLY")
else (SBML_FOUND AND HOPS_SBML_SUPPORT)
    message(STATUS "SBML not found, continuing without.")
endif (SBML_FOUND AND HOPS_SBML_SUPPORT)

find_package(HDF5 COMPONENTS C CXX)
if (HDF5_FOUND AND HOPS_HDF5_SUPPORT)
    # The linking and finding of HDF5 is done in the FileWriter CMakeLists.txt
    if (NOT HOPS_LIBRARY_TYPE STREQUAL "HEADER_ONLY")
        target_sources(hops PRIVATE Hdf5Reader.cpp)
    endif (NOT HOPS_LIBRARY_TYPE STREQUAL "HEADER_ONLY")
else (HDF5_FOUND AND HOPS_HDF5_SUPPORT)
    message("NOT ADDING sources for HdfReader.cpp")
endif (HDF5_FOUND AND HOPS_HDF5_SUPPORT)
