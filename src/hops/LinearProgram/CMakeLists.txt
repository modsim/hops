if (NOT HOPS_LIBRARY_TYPE STREQUAL "HEADER_ONLY")
    target_sources(hops PRIVATE
            GurobiEnvironmentSingleton.hpp
            LinearProgram.hpp
            LinearProgramClpImpl.hpp
            LinearProgramFactory.hpp
            LinearProgramGurobiImpl.hpp
            LinearProgramSolution.hpp
            LinearProgramStatus.hpp
            )
endif (NOT HOPS_LIBRARY_TYPE STREQUAL "HEADER_ONLY")

set(GUROBI_ENABLE ON)
find_package(GUROBI)
if (GUROBI_FOUND)
    target_compile_definitions(hops INTERFACE HOPS_GUROBI_FOUND)
    target_compile_definitions(hops ${SCOPE} HOPS_GUROBI_FOUND)
    target_include_directories(hops INTERFACE ${GUROBI_INCLUDE_DIR})
    target_include_directories(hops ${SCOPE} ${GUROBI_INCLUDE_DIR})
    # IMPORTANT: Gurobi C++ Libraries have to be listed before Gurobi C Libraries to prevent linker error
    target_link_libraries(hops INTERFACE ${GUROBI_CXX_LIBRARY} ${GUROBI_LIBRARY})
    target_link_libraries(hops ${SCOPE} ${GUROBI_CXX_LIBRARY} ${GUROBI_LIBRARY})
    if (NOT HOPS_LIBRARY_TYPE STREQUAL "HEADER_ONLY")
        target_sources(hops PRIVATE GurobiEnvironmentSingleton.cpp LinearProgramGurobiImpl.cpp)
    endif (NOT HOPS_LIBRARY_TYPE STREQUAL "HEADER_ONLY")
endif ()

find_package(CLP)
if (CLP_FOUND)
    target_compile_definitions(hops INTERFACE HOPS_CLP_FOUND)
    target_compile_definitions(hops ${SCOPE} HOPS_CLP_FOUND)
    target_include_directories(hops INTERFACE ${CLP_INCLUDE_DIRS})
    target_include_directories(hops ${SCOPE} ${CLP_INCLUDE_DIRS})
    target_link_libraries(hops INTERFACE ${CLP_LIBRARIES})
    target_link_libraries(hops ${SCOPE} ${CLP_LIBRARIES})
    if (NOT HOPS_LIBRARY_TYPE STREQUAL "HEADER_ONLY")
        target_sources(hops PRIVATE LinearProgramClpImpl.cpp)
    endif (NOT HOPS_LIBRARY_TYPE STREQUAL "HEADER_ONLY")
endif ()

if (NOT GUROBI_FOUND AND NOT CLP_FOUND)
    message(WARNING "No LP solver configured. This is okay, if you do not require linear programming. Otherwise install gurobi or CLP in order to use the LinearProgram class.")
endif ()
