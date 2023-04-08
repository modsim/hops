if (NOT HOPS_LIBRARY_TYPE STREQUAL "HEADER_ONLY")
    target_sources(hops PRIVATE
            AcceptanceRateTarget.hpp
            AcceptanceRateTarget.cpp
            AcceptanceRateTuner.hpp
            AcceptanceRateTuner.cpp
            BinarySearchAcceptanceRateTuner.hpp
            BinarySearchAcceptanceRateTuner.cpp
            ExpectedSquaredJumpDistanceTarget.hpp
            ExpectedSquaredJumpDistanceTuner.hpp
            ExpectedSquaredJumpDistanceTuner.cpp
            ThompsonSamplingTuner.hpp
            ThompsonSamplingTuner.cpp
            )
endif (NOT HOPS_LIBRARY_TYPE STREQUAL "HEADER_ONLY")
