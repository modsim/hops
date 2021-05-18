if (NOT HOPS_LIBRARY_TYPE STREQUAL "HEADER_ONLY")
    target_sources(hops PRIVATE
            IsAcceptProposalAvailable.hpp
            IsCalculateLogAcceptanceProbabilityAvailable.hpp
            MetropolisHastingsFilter.hpp
            NoOpDrawAdapter.hpp
            )
endif (NOT HOPS_LIBRARY_TYPE STREQUAL "HEADER_ONLY")
