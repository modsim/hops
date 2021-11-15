if (NOT HOPS_LIBRARY_TYPE STREQUAL "HEADER_ONLY")
    target_sources(hops PRIVATE
            DNest4Adapter.hpp
            DNest4EnvironmentSingleton.hpp
            )
elseif (NOT HOPS_LIBRARY_TYPE STREQUAL "HEADER_ONLY")
    target_link_libraries(hops INTERFACE dnest4)
endif (NOT HOPS_LIBRARY_TYPE STREQUAL "HEADER_ONLY")
