#ifndef HOPS_MODELMIXIN_HPP
#define HOPS_MODELMIXIN_HPP

#include <cmath>
#include <hops/MarkovChain/Draw/IsCalculateLogAcceptanceProbabilityAvailable.hpp>
#include <hops/MarkovChain/Recorder/IsStoreMetropolisHastingsInfoRecordAvailable.hpp>
#include <hops/RandomNumberGenerator/RandomNumberGenerator.hpp>

namespace hops {
    /**
     * @brief ModelMixin Mixin to add model likelihood to calculateLogAcceptanceRate().
     * @details Useful for MarkovChainProposer classes, that do not already contain the model.
     * @tparam MarkovChainProposer
     * @tparam ModelImpl
     */
    template<typename MarkovChainProposer, typename ModelImpl>
    class ModelMixin : public MarkovChainProposer, public ModelImpl {
    public:
        ModelMixin(const MarkovChainProposer &markovChainProposer, const ModelImpl &model) :
                MarkovChainProposer(markovChainProposer),
                ModelImpl(model) {
            proposalNegativeLogLikelihood = 0;
            stateNegativeLogLikelihood = ModelImpl::calculateNegativeLogLikelihood(MarkovChainProposer::getState());
        }

        void acceptProposal();

        double calculateLogAcceptanceProbability();

        double getNegativeLogLikelihoodOfCurrentState();

    private:
        double stateNegativeLogLikelihood;
        double proposalNegativeLogLikelihood;
    };

    template<typename MarkovChainProposer, typename ModelImpl>
    void ModelMixin<MarkovChainProposer, ModelImpl>::acceptProposal() {
        MarkovChainProposer::acceptProposal();
        stateNegativeLogLikelihood = proposalNegativeLogLikelihood;
    }

    template<typename MarkovChainProposer, typename ModelImpl>
    double ModelMixin<MarkovChainProposer, ModelImpl>::calculateLogAcceptanceProbability() {
        double acceptanceProbability = 0;
        if constexpr(IsCalculateLogAcceptanceProbabilityAvailable<MarkovChainProposer>::value) {
            acceptanceProbability += MarkovChainProposer::calculateLogAcceptanceProbability();
        }
        if (std::isfinite(acceptanceProbability)) {
            proposalNegativeLogLikelihood = ModelImpl::calculateNegativeLogLikelihood(
                    MarkovChainProposer::getProposal());
            acceptanceProbability += stateNegativeLogLikelihood - proposalNegativeLogLikelihood;
            if constexpr(IsStoreMetropolisHastingsInfoRecordAvailable<MarkovChainProposer>::value) {
                MarkovChainProposer::storeMetropolisHastingsInfoRecord("likelihood");
            } else if constexpr(IsStoreMetropolisHastingsInfoRecordAvailable<ModelImpl>::value) {
                ModelImpl::storeMetropolisHastingsInfoRecord("likelihood");
            }
        } else {
            if constexpr(IsStoreMetropolisHastingsInfoRecordAvailable<MarkovChainProposer>::value) {
                MarkovChainProposer::storeMetropolisHastingsInfoRecord("polytope");
            } else if constexpr(IsStoreMetropolisHastingsInfoRecordAvailable<ModelImpl>::value) {
                ModelImpl::storeMetropolisHastingsInfoRecord("polytope");
            }
        }
        return acceptanceProbability;
    }

    template<typename MarkovChainProposer, typename ModelImpl>
    double ModelMixin<MarkovChainProposer, ModelImpl>::getNegativeLogLikelihoodOfCurrentState() {
        return stateNegativeLogLikelihood;
    }
}

#endif //HOPS_MODELMIXIN_HPP
