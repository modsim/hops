target_sources(hops PRIVATE CsvReader.cpp)

set(sbml_DIR /usr/local/lib/cmake CACHE PATH "Path to sbml-config.cmake. Usually found in /usr/local/cmake or /usr/local/lib/cmake")

find_package(sbml CONFIG REQUIRED)

target_link_libraries(hops PUBLIC sbml)
target_sources(hops PRIVATE SbmlReader.cpp)
