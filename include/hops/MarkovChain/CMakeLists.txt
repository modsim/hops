target_sources(hops PRIVATE
        MarkovChain.hpp
        MarkovChainAdapter.hpp
        MarkovChainFactory.hpp
        MarkovChainType.hpp
        StateTransformation.hpp
        )

add_subdirectory(Draw)
add_subdirectory(Proposal)
add_subdirectory(Recorder)
