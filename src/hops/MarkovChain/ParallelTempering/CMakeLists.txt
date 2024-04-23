if (HOPS_MPI)
    find_package(MPI)

    if (MPI_FOUND)
        # See Parallel directory for more on MPI

        if (NOT HOPS_LIBRARY_TYPE STREQUAL "HEADER_ONLY")
            target_sources(hops PRIVATE
                    ParallelTempering.hpp
            )
        endif (NOT HOPS_LIBRARY_TYPE STREQUAL "HEADER_ONLY")

    else (NOT MPI_FOUND)
        message(STATUS "MPI could not be found or was not configured for HOPS. HOPS continuing without MPI support.")
    endif (MPI_FOUND)
endif (HOPS_MPI)
