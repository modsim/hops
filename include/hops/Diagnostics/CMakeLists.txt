if (NOT HOPS_LIBRARY_TYPE STREQUAL "HEADER_ONLY")
    target_sources(hops PRIVATE
            EffectiveSampleSize.hpp
            ExpectedSquaredJumpDistance.hpp
            PotentialScaleReductionFactor.hpp
            )
endif (NOT HOPS_LIBRARY_TYPE STREQUAL "HEADER_ONLY")
