#ifndef HOPS_MODELMIXIN_HPP
#define HOPS_MODELMIXIN_HPP

#include <cmath>

#include <hops/MarkovChain/Draw/IsCalculateLogAcceptanceProbabilityAvailable.hpp>
#include <hops/RandomNumberGenerator/RandomNumberGenerator.hpp>

namespace hops {
    /**
     * @brief ModelMixin Mixin to add model likelihood to computeLogAcceptanceRate().
     * @details Useful for MarkovChainProposer classes, that do not already contain the model.
     * @tparam MarkovChainProposer
     * @tparam ModelImpl
     */
    template<typename MarkovChainProposer, typename ModelType>
    class ModelMixin : public MarkovChainProposer, public ModelType {
    public:
        ModelMixin(const MarkovChainProposer &markovChainProposer, const ModelType &model) :
                MarkovChainProposer(markovChainProposer),
                ModelType(model) {
            proposalNegativeLogLikelihood = 0;
            stateNegativeLogLikelihood = ModelType::computeNegativeLogLikelihood(MarkovChainProposer::getState());
        }

        void acceptProposal();

        double computeLogAcceptanceProbability();

        double getNegativeLogLikelihoodOfCurrentState();

    private:
        double stateNegativeLogLikelihood;
        double proposalNegativeLogLikelihood;
    };

    template<typename MarkovChainProposer, typename ModelType>
    void ModelMixin<MarkovChainProposer, ModelType>::acceptProposal() {
        MarkovChainProposer::acceptProposal();
        stateNegativeLogLikelihood = proposalNegativeLogLikelihood;
    }

    template<typename MarkovChainProposer, typename ModelType>
    double ModelMixin<MarkovChainProposer, ModelType>::computeLogAcceptanceProbability() {
        double acceptanceProbability = 0;
        if constexpr(IsCalculateLogAcceptanceProbabilityAvailable<MarkovChainProposer>::value) {
            acceptanceProbability += MarkovChainProposer::computeLogAcceptanceProbability();
        }
        if (std::isfinite(acceptanceProbability)) {
            proposalNegativeLogLikelihood = ModelType::computeNegativeLogLikelihood(
                    MarkovChainProposer::getProposal());
            acceptanceProbability += stateNegativeLogLikelihood - proposalNegativeLogLikelihood;
        }
        return acceptanceProbability;
    }

    template<typename MarkovChainProposer, typename ModelType>
    double ModelMixin<MarkovChainProposer, ModelType>::getNegativeLogLikelihoodOfCurrentState() {
        return stateNegativeLogLikelihood;
    }
}

#endif //HOPS_MODELMIXIN_HPP