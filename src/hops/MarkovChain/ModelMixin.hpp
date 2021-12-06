#ifndef HOPS_MODELMIXIN_HPP
#define HOPS_MODELMIXIN_HPP

#include <cmath>

#include <hops/MarkovChain/Draw/IsCalculateLogAcceptanceProbabilityAvailable.hpp>
#include <hops/RandomNumberGenerator/RandomNumberGenerator.hpp>
#include <hops/Utility/VectorType.hpp>

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

        VectorType& acceptProposal();

        double computeLogAcceptanceProbability();

        double getStateNegativeLogLikelihood();

        double getProposalNegativeLogLikelihood();

        void setState(const VectorType&);

    private:
        double stateNegativeLogLikelihood;
        double proposalNegativeLogLikelihood;
    };

    template<typename MarkovChainProposer, typename ModelType>
    VectorType& ModelMixin<MarkovChainProposer, ModelType>::acceptProposal() {
        stateNegativeLogLikelihood = proposalNegativeLogLikelihood;
        return MarkovChainProposer::acceptProposal();
    }

    template<typename MarkovChainProposer, typename ModelType>
    double ModelMixin<MarkovChainProposer, ModelType>::computeLogAcceptanceProbability() {
        double acceptanceProbability = MarkovChainProposer::computeLogAcceptanceProbability();

        if (std::isfinite(acceptanceProbability)) {
            proposalNegativeLogLikelihood = ModelType::computeNegativeLogLikelihood(
                    MarkovChainProposer::getProposal());
            acceptanceProbability += stateNegativeLogLikelihood - proposalNegativeLogLikelihood;
        }

        return acceptanceProbability;
    }

    template<typename MarkovChainProposer, typename ModelType>
    double ModelMixin<MarkovChainProposer, ModelType>::getStateNegativeLogLikelihood() {
        return stateNegativeLogLikelihood + MarkovChainProposer::getStateNegativeLogLikelihood();
    }

    template<typename MarkovChainProposer, typename ModelType>
    double ModelMixin<MarkovChainProposer, ModelType>::getProposalNegativeLogLikelihood() {
        return proposalNegativeLogLikelihood + MarkovChainProposer::getProposalNegativeLogLikelihood();
    }

    template<typename MarkovChainProposer, typename ModelType>
    void ModelMixin<MarkovChainProposer, ModelType>::setState(const VectorType& state) {
        MarkovChainProposer::setState(state);
        stateNegativeLogLikelihood = ModelType::computeNegativeLogLikelihood(state);
    }
}

#endif //HOPS_MODELMIXIN_HPP
