target_sources(hops PRIVATE
        IsSetStepSizeAvailable.hpp
        MarkovChain.hpp
        MarkovChainAdapter.hpp
        MarkovChainFactory.hpp
        MarkovChainType.hpp
        StateTransformation.hpp
        )

add_subdirectory(Draw)
add_subdirectory(ParallelTempering)
add_subdirectory(Proposal)
add_subdirectory(Recorder)
