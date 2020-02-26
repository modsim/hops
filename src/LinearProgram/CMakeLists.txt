# TODO switch should probably be set somewhere else or by env
set(GUROBI_ENABLE ON)

find_package(GUROBI)
if (GUROBI_FOUND)
    add_definitions(-DHOPS_GUROBI_FOUND)
endif ()

target_include_directories(HOPS PUBLIC
        ${GUROBI_INCLUDE_DIR}
        )
target_link_libraries(HOPS PRIVATE
        # IMPORTANT: Gurobi C++ Libraries have to be listed before Gurobi C Libraries to prevent linker error
        ${GUROBI_CXX_LIBRARY}
        ${GUROBI_LIBRARY}
        )

target_sources(HOPS PRIVATE
        GurobiEnvironmentSingleton.cpp
        LinearProgram.cpp
        LinearProgramGurobiImpl.cpp
        )
