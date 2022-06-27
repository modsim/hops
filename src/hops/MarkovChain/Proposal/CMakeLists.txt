if (NOT HOPS_LIBRARY_TYPE STREQUAL "HEADER_ONLY")
    target_sources(hops PRIVATE
            AdaptiveMetropolisProposal.hpp
            BallWalkProposal.hpp
            BilliardAdaptiveMetropolisProposal.hpp
            BilliardMALAProposal.hpp
            ChordStepDistributions.hpp
            CoordinateHitAndRunProposal.hpp
            CSmMALAProposal.hpp
            DikinEllipsoidCalculator.hpp
            DikinProposal.hpp
            GaussianProposal.hpp
            HitAndRunProposal.hpp
            TruncatedGaussianProposal.hpp
            Proposal.hpp
            ProposalFactory.hpp
            ProposalParameter.hpp
            ProposalStatistics.hpp
            ProposalStatistics.cpp
            # TODO
#            ReversibleJumpProposalOld.hpp
#            ReversibleJumpProposal.hpp
            Reflector.hpp
            BilliardMALAProposal.hpp
            TruncatedNormalDistribution.hpp
            )
endif (NOT HOPS_LIBRARY_TYPE STREQUAL "HEADER_ONLY")
