find_package(MPI)

if (MPI_FOUND)
    # See Parallel directory for more on MPI

    if (NOT HOPS_LIBRARY_TYPE STREQUAL "HEADER_ONLY")
        target_sources(hops PRIVATE
                Coldness.hpp
                ParallelTempering.hpp
                )
    endif (NOT HOPS_LIBRARY_TYPE STREQUAL "HEADER_ONLY")

else (MPI_FOUND)
    message(WARNING "MPI could not be found. Continuing without MPI support.")
endif (MPI_FOUND)
