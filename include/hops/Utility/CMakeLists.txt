if (NOT HOPS_LIBRARY_TYPE STREQUAL "HEADER_ONLY")
    target_sources(hops PRIVATE
            ChainData.hpp
            Data.hpp
            Data.cpp
            Exceptions.hpp
            Problem.hpp
            Run.hpp
            )
endif (NOT HOPS_LIBRARY_TYPE STREQUAL "HEADER_ONLY")
