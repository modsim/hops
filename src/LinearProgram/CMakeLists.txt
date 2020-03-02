set(GUROBI_ENABLE ON)

find_package(GUROBI)
find_package(CLP)

if (GUROBI_FOUND)
    target_compile_definitions(HOPS PUBLIC HOPS_GUROBI_FOUND)
    target_include_directories(HOPS PUBLIC ${GUROBI_INCLUDE_DIR})
    # IMPORTANT: Gurobi C++ Libraries have to be listed before Gurobi C Libraries to prevent linker error
    target_link_libraries(HOPS PRIVATE ${GUROBI_CXX_LIBRARY} ${GUROBI_LIBRARY})
    target_sources(HOPS PRIVATE GurobiEnvironmentSingleton.cpp LinearProgramGurobiImpl.cpp)
endif ()

if (CLP_FOUND)
    target_compile_definitions(HOPS PUBLIC HOPS_CLP_FOUND)
    target_include_directories(HOPS PUBLIC ${CLP_INCLUDE_DIRS})
    target_link_libraries(HOPS PRIVATE ${CLP_LIBRARIES})
    target_sources(HOPS PRIVATE LinearProgramClpImpl.cpp)
endif ()

if (NOT GUROBI_FOUND AND NOT CLP_FOUND)
    message(WARNING "No LP solver configured. This is okay, if you do not require linear programming. Otherwise install gurobi or CLP in order to use the LinearProgram class.")
endif ()


target_sources(HOPS PRIVATE LinearProgram.cpp)
