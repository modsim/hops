if (NOT HOPS_LIBRARY_TYPE STREQUAL "HEADER_ONLY")
    target_sources(hops PRIVATE
            AcceptanceRateTarget.hpp
            AcceptanceRateTuner.hpp
            AcceptanceRateTuner.cpp
            BinarySearchAcceptanceRateTuner.hpp
            BinarySearchAcceptanceRateTuner.cpp
            ExpectedSquaredJumpDistanceTarget.hpp
            ExpectedSquaredJumpDistanceTuner.hpp
            ExpectedSquaredJumpDistanceTuner.cpp
            #SimpleExpectedSquaredJumpDistanceTuner.hpp
            #SimpleExpectedSquaredJumpDistanceTuner.cpp
            ThompsonSamplingTuner.hpp
            )
endif (NOT HOPS_LIBRARY_TYPE STREQUAL "HEADER_ONLY")
