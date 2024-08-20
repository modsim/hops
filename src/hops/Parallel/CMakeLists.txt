if (HOPS_MPI)
    find_package(MPI)

    if (MPI_FOUND)
        message(STATUS "found MPI ${MPI_LIBRARIES} ${MPI_INCLUDE_PATH} ${MPI_C_INCLUDE_DIRS} ${MPI_C_LIBRARIES}.")

        target_compile_definitions(hops INTERFACE HOPS_MPI_SUPPORTED)
        target_compile_definitions(hops ${SCOPE} HOPS_MPI_SUPPORTED)
        target_include_directories(hops INTERFACE ${MPI_C_INCLUDE_DIRS})
        target_include_directories(hops ${SCOPE} ${MPI_C_INCLUDE_DIRS})
        target_link_libraries(hops INTERFACE ${MPI_C_LIBRARIES})
        target_link_libraries(hops ${SCOPE} ${MPI_C_LIBRARIES})

        if (NOT HOPS_LIBRARY_TYPE STREQUAL "HEADER_ONLY")
            target_sources(hops PRIVATE MpiInitializerFinalizer.hpp)
        endif (NOT HOPS_LIBRARY_TYPE STREQUAL "HEADER_ONLY")

    else (MPI_FOUND)
        message(WARNING "MPI could not be found. Continuing without MPI support.")
    endif (MPI_FOUND)
endif (HOPS_MPI)
