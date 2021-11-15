if (NOT HOPS_LIBRARY_TYPE STREQUAL "HEADER_ONLY")
    target_sources(hops PRIVATE
            MaximumVolumeEllipsoid.hpp
            MaximumVolumeEllipsoid.cpp
            NormalizePolytope.hpp
            SimplexFactory.hpp
            )
endif (NOT HOPS_LIBRARY_TYPE STREQUAL "HEADER_ONLY")
