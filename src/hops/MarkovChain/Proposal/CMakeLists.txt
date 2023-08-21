if (NOT HOPS_LIBRARY_TYPE STREQUAL "HEADER_ONLY")
    target_sources(hops PRIVATE
            AdaptiveMetropolisProposal.hpp
            BallWalkProposal.hpp
            BilliardAdaptiveMetropolisProposal.hpp
            BilliardMALAProposal.hpp
            BilliardWalkProposal.hpp
            ChordStepDistributions.hpp
            CoordinateHitAndRunProposal.hpp
            CSmMALAProposal.hpp
            DikinEllipsoidCalculator.hpp
            DikinProposal.hpp
            GaussianProposal.hpp
            HitAndRunProposal.hpp
            IsGetStepSizeAvailable.hpp
            IsSetStepSizeAvailable.hpp
            Proposal.hpp
            ProposalFactory.hpp
            ProposalParameter.hpp
            ReversibleJumpProposal.hpp
            ReversibleJumpProposal.cpp
            Reflector.hpp
            TruncatedGaussianProposal.hpp
            TruncatedNormalDistribution.hpp
            )
endif (NOT HOPS_LIBRARY_TYPE STREQUAL "HEADER_ONLY")
