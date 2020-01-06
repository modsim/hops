target_sources(nups PUBLIC
        MarkovChain.hpp
        MarkovChainAdapter.hpp
        MarkovChainFactory.hpp
        MarkovChainType.hpp
        StateTransformation.hpp
        )

add_subdirectory(Draw)
add_subdirectory(Proposal)
add_subdirectory(Recorder)
