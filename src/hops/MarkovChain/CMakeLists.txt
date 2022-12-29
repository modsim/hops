if (NOT HOPS_LIBRARY_TYPE STREQUAL "HEADER_ONLY")
    target_sources(hops PRIVATE
            MarkovChain.hpp
            MarkovChainAdapter.hpp
            MarkovChainFactory.hpp
            MarkovChainType.hpp
            MarkovChainType.cpp
            ModelMixin.hpp
            ModelWrapper.hpp
            StateTransformation.hpp
            )
endif (NOT HOPS_LIBRARY_TYPE STREQUAL "HEADER_ONLY")

add_subdirectory(Draw)
add_subdirectory(ParallelTempering)
add_subdirectory(Proposal)
add_subdirectory(Recorder)
add_subdirectory(Tuning)

