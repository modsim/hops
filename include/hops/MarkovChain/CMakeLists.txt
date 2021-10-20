if (NOT HOPS_LIBRARY_TYPE STREQUAL "HEADER_ONLY")
    target_sources(hops PRIVATE
            IsGetColdnessAvailable.hpp
            IsGetExchangeAttemptProbabilityAvailable.hpp
            IsGetStepSizeAvailable.hpp
            IsSetColdnessAvailable.hpp
            IsSetExchangeAttemptProbabilityAvailable.hpp
            IsSetStepSizeAvailable.hpp
            MarkovChain.hpp
            MarkovChainAdapter.hpp
            MarkovChainAttribute.hpp
            MarkovChainFactory.hpp
            MarkovChainType.hpp
            StateTransformation.hpp
            )
endif (NOT HOPS_LIBRARY_TYPE STREQUAL "HEADER_ONLY")

add_subdirectory(Draw)
add_subdirectory(ParallelTempering)
add_subdirectory(Proposal)
add_subdirectory(Recorder)
add_subdirectory(ReversibleJump)
add_subdirectory(Tuning)

