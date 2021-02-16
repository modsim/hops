target_sources(hops PRIVATE CsvReader.cpp Hdf5Reader.cpp)

find_package(SBML)

if (SBML_FOUND)
    MESSAGE(STATUS "SBML found")
    target_link_libraries(hops PUBLIC sbml)
    target_sources(hops PRIVATE SbmlReader.cpp)
else ()
    MESSAGE(STATUS "SBML not found, continuing without.")
endif ()

find_package(HDF5 COMPONENTS C CXX)

if (HDF5_FOUND)
    target_sources(hops PRIVATE Hdf5Reader.cpp)
endif ()
