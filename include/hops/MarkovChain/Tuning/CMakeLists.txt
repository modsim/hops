if (NOT HOPS_LIBRARY_TYPE STREQUAL "HEADER_ONLY")
    target_sources(hops PRIVATE
            AcceptanceRateTuner.hpp
            AcceptanceRateTuner.cpp
            BinarySearchAcceptanceRateTuner.hpp
            BinarySearchAcceptanceRateTuner.cpp
            ExpectedSquaredJumpDistanceTuner.hpp
            ExpectedSquaredJumpDistanceTuner.cpp
            SimpleExpectedSquaredJumpDistanceTuner.hpp
            SimpleExpectedSquaredJumpDistanceTuner.cpp
            )
endif (NOT HOPS_LIBRARY_TYPE STREQUAL "HEADER_ONLY")
