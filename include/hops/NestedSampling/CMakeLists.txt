if (NOT HOPS_LIBRARY_TYPE STREQUAL "HEADER_ONLY")
    find_package(DNest4)

    if (DNest4_FOUND)
        message(STATUS "Found DNest4 at ${DNest4_INCLUDE_DIR} (includes), ${DNest4_LIBRARY} (library)")
        target_sources(hops PRIVATE
                DNest4Adapter.hpp
                DNest4EnvironmentSingleton.hpp
                )
        target_include_directories(hops PRIVATE ${DNest4_INCLUDE_DIR})
        target_link_libraries(hops PUBLIC ${DNest4_LIBRARY})
    else (DNest4_FOUND)
        message(WARNING "Did not find DNest4 on this system. The Nested sampling feature will not be built.")
    endif (DNest4_FOUND)

endif (NOT HOPS_LIBRARY_TYPE STREQUAL "HEADER_ONLY")
