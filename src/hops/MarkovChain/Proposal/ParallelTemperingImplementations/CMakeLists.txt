if (NOT HOPS_LIBRARY_TYPE STREQUAL "HEADER_ONLY")
    target_sources(hops PRIVATE
            ParallelTempering.hpp
            ParallelTemperingSEOBoostInterprocess.cpp
            ParallelTemperingSEOBoostInterprocess.hpp
    )
endif (NOT HOPS_LIBRARY_TYPE STREQUAL "HEADER_ONLY")
