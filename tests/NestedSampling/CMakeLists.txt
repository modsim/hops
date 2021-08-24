if (NOT HOPS_LIBRARY_TYPE STREQUAL "HEADER_ONLY")
    find_package(DNest4)

    if (DNest4_FOUND)
        message(STATUS "Building nested sampling tests because DNest4 was found")
    elseif (DNest4_FOUND)
        message(WARNING "Not building nested sampling tests because DNest4 was not found")
    endif (DNest4_FOUND)

endif (NOT HOPS_LIBRARY_TYPE STREQUAL "HEADER_ONLY")
