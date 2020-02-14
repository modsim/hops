#ifndef HOPS_MARKOVCHAINFACTORY_HPP
#define HOPS_MARKOVCHAINFACTORY_HPP

#include <hops/MarkovChain/MarkovChain.hpp>
#include <hops/MarkovChain/MarkovChainType.hpp>
#include <hops/MarkovChain/Proposal/CoordinateHitAndRunProposal.hpp>
#include <hops/MarkovChain/Proposal/CoordinateHitAndRunProposal.hpp>
#include <hops/MarkovChain/Proposal/CoordinateHitAndRunRoundedProposal.hpp>
#include <hops/MarkovChain/MarkovChainAdapter.hpp>
#include <hops/MarkovChain/Draw/NoOpDraw.hpp>
#include <hops/MarkovChain/Recorder/StateRecorder.hpp>
#include <hops/MarkovChain/StateTransformation.hpp>
#include <hops/MarkovChain/Recorder/TimestampRecorder.hpp>
#include <hops/Transformation/Transformation.hpp>

namespace hops {
    class MarkovChainFactory {
    public:
        /**
         * @tparam StateSpace Class or Struct containing at least public members A, b, and startingPoint.
         *                    Other required public members depend on selected featureFlags.
         * @param stateSpace The state space to sample, in general a polytope with likelihood function.
         * @param proposalType Type of proposal distribution.
         * @param featureFlags Mask of additional features for the Markov chain.
         * @return
         */
        template<typename StateSpace>
        static std::unique_ptr<hops::MarkovChain>
        createMarkovChain(const StateSpace &stateSpace, MarkovChainType proposalType);
    };

    template<typename StateSpace>
    std::unique_ptr<hops::MarkovChain>
    MarkovChainFactory::createMarkovChain(const StateSpace &stateSpace, MarkovChainType proposalType) {
        switch (proposalType) {
            case MarkovChainType::CoordinateHitAndRun: {
                return std::unique_ptr<MarkovChain>(
                        new MarkovChainAdapter(
                                StateRecorder(
                                        NoOpDraw(
                                                CoordinateHitAndRunProposal<decltype(stateSpace.A), decltype(stateSpace.b)>(
                                                        stateSpace.A,
                                                        stateSpace.b,
                                                        stateSpace.startingPoint)
                                        )
                                )
                        )
                );
            }
            case MarkovChainType::CoordinateHitAndRunRoundedProposals: {
                return std::unique_ptr<MarkovChain>(
                        new MarkovChainAdapter(
                                StateRecorder(
                                        StateTransformation(
                                                NoOpDraw(
                                                        CoordinateHitAndRunRoundedProposal<decltype(stateSpace.A), decltype(stateSpace.b)>(
                                                                stateSpace.A,
                                                                stateSpace.roundedT,
                                                                stateSpace.b,
                                                                stateSpace.startingPoint)
                                                ),
                                                Transformation(stateSpace.N, stateSpace.shift)
                                        )
                                )
                        )
                );
            }
            case MarkovChainType::CoordinateHitAndRunRoundedStateSpace: {
                return std::unique_ptr<MarkovChain>(
                        new MarkovChainAdapter(
                                StateRecorder(
                                        StateTransformation(
                                                NoOpDraw(
                                                        CoordinateHitAndRunProposal<decltype(stateSpace.roundedA), decltype(stateSpace.roundedb)>(
                                                                stateSpace.roundedA,
                                                                stateSpace.roundedb,
                                                                stateSpace.roundedStartingPoint)
                                                ),
                                                Transformation(stateSpace.roundedN, stateSpace.roundedShift)
                                        )
                                )
                        )
                );
            }
            default: {
                throw std::runtime_error("Error");
            }
        }
    }
}

#endif //HOPS_MARKOVCHAINFACTORY_HPP
