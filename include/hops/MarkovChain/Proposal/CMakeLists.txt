if (NOT HOPS_LIBRARY_TYPE STREQUAL "HEADER_ONLY")
    target_sources(hops PRIVATE
            AdaptiveMetropolisProposal.hpp
            BallWalkProposal.hpp
            ChordStepDistributions.hpp
            CoordinateHitAndRunProposal.hpp
            DikinEllipsoidCalculator.hpp
            DikinProposal.hpp
            GaussianProposal.hpp
            HitAndRunProposal.hpp
            TruncatedNormalDistribution.hpp
            )
endif (NOT HOPS_LIBRARY_TYPE STREQUAL "HEADER_ONLY")
