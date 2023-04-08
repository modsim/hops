find_package(MPI)

if (MPI_FOUND)
    target_compile_definitions(hops INTERFACE HOPS_MPI_SUPPORTED)
    target_compile_definitions(hops ${SCOPE} HOPS_MPI_SUPPORTED)
    target_include_directories(hops INTERFACE ${MPI_INCLUDE_PATH})
    target_include_directories(hops ${SCOPE} ${MPI_INCLUDE_PATH})
    target_link_libraries(hops INTERFACE ${MPI_LIBRARIES})
    target_link_libraries(hops ${SCOPE} ${MPI_LIBRARIES})

    if (NOT HOPS_LIBRARY_TYPE STREQUAL "HEADER_ONLY")
        target_sources(hops PRIVATE MpiInitializerFinalizer.hpp)
    endif (NOT HOPS_LIBRARY_TYPE STREQUAL "HEADER_ONLY")

else (MPI_FOUND)
    message(WARNING "MPI could not be found. Continuing without MPI support.")
endif (MPI_FOUND)
