target_sources(hops PRIVATE
        AcceptanceRateTuner.hpp
        IsGetColdnessAvailable.hpp
        IsGetExchangeAttemptProbabilityAvailable.hpp
        IsGetStepSizeAvailable.hpp
        IsResetAcceptanceRateAvailable.hpp
        IsSetColdnessAvailable.hpp
        IsSetExchangeAttemptProbabilityAvailable.hpp
        IsSetFisherWeightAvailable.hpp
        IsSetStepSizeAvailable.hpp
        MarkovChain.hpp
        MarkovChainAdapter.hpp
        MarkovChainAttribute.hpp
        MarkovChainFactory.hpp
        MarkovChainType.hpp
        StateTransformation.hpp
        )

add_subdirectory(Draw)
add_subdirectory(ParallelTempering)
add_subdirectory(Proposal)
add_subdirectory(Recorder)
