#ifndef NUPS_MARKOVCHAINFACTORY_HPP
#define NUPS_MARKOVCHAINFACTORY_HPP

#include <nups/MarkovChain/MarkovChain.hpp>
#include <nups/MarkovChain/MarkovChainType.hpp>
#include <nups/MarkovChain/Proposal/CoordinateHitAndRunProposal.hpp>
#include <nups/MarkovChain/Proposal/CoordinateHitAndRunProposal.hpp>
#include <nups/MarkovChain/Proposal/CoordinateHitAndRunRoundedProposal.hpp>
#include <nups/MarkovChain/MarkovChainAdapter.hpp>
#include <nups/MarkovChain/Draw/NoOpDraw.hpp>
#include <nups/MarkovChain/Recorder/StateRecorder.hpp>
#include <nups/MarkovChain/StateTransformation.hpp>
#include <nups/MarkovChain/Recorder/TimestampRecorder.hpp>
#include <nups/Transformation/Transformation.hpp>

namespace nups {
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
        static std::unique_ptr<nups::MarkovChain>
        createMarkovChain(const StateSpace &stateSpace, MarkovChainType proposalType);
    };

    template<typename StateSpace>
    std::unique_ptr<nups::MarkovChain>
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

#endif //NUPS_MARKOVCHAINFACTORY_HPP
