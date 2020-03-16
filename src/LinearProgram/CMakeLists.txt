set(GUROBI_ENABLE ON)

find_package(GUROBI)
find_package(CLP)

if (GUROBI_FOUND)
    target_compile_definitions(hops PUBLIC HOPS_GUROBI_FOUND)
    target_include_directories(hops PUBLIC ${GUROBI_INCLUDE_DIR})
    # IMPORTANT: Gurobi C++ Libraries have to be listed before Gurobi C Libraries to prevent linker error
    # TODO replace public by interface?
    target_link_libraries(hops PUBLIC ${GUROBI_CXX_LIBRARY} ${GUROBI_LIBRARY})
    target_sources(hops PRIVATE GurobiEnvironmentSingleton.cpp LinearProgramGurobiImpl.cpp)
endif ()

if (CLP_FOUND)
    target_compile_definitions(hops PUBLIC HOPS_CLP_FOUND)
    target_include_directories(hops PUBLIC ${CLP_INCLUDE_DIRS})
    # TODO replace public by interface?
    target_link_libraries(hops PUBLIC ${CLP_LIBRARIES})
    target_sources(hops PRIVATE LinearProgramClpImpl.cpp)
endif ()

if (NOT GUROBI_FOUND AND NOT CLP_FOUND)
    message(WARNING "No LP solver configured. This is okay, if you do not require linear programming. Otherwise install gurobi or CLP in order to use the LinearProgram class.")
endif ()


target_sources(hops PRIVATE LinearProgram.cpp)
