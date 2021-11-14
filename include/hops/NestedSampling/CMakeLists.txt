if (NOT HOPS_LIBRARY_TYPE STREQUAL "HEADER_ONLY")
    target_include_directories(hops PUBLIC
            $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/extern/DNest4/code>
            $<INSTALL_INTERFACE:include/>  # <prefix>/include/mylib
    )
    target_link_libraries(hops PUBLIC dnest4)

    target_sources(hops PRIVATE
            DNest4Adapter.hpp
            DNest4EnvironmentSingleton.hpp
            )


elseif (NOT HOPS_LIBRARY_TYPE STREQUAL "HEADER_ONLY")
    target_link_libraries(hops INTERFACE dnest4)
endif (NOT HOPS_LIBRARY_TYPE STREQUAL "HEADER_ONLY")
