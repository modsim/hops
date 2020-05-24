target_sources(hops PRIVATE CsvReader.cpp)

set(sbml_DIR /usr/local/lib/cmake CACHE PATH "Path to sbml-config.cmake. Usually found in /usr/local/cmake or /usr/local/lib/cmake on linux")

find_package(sbml CONFIG)

if(${SBML_FOUND}) 
	target_link_libraries(hops PUBLIC sbml)
	target_sources(hops PRIVATE SbmlReader.cpp)
endif()

