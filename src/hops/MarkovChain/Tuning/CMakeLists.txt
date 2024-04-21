if (HOPS_TUNING)
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
endif (HOPS_TUNING)
