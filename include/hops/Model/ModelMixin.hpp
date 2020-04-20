#ifndef HOPS_MODELMIXIN_HPP
#define HOPS_MODELMIXIN_HPP

#include <hops/MarkovChain/Draw/IsCalculateLogAcceptanceProbabilityAvailable.hpp>
#include <hops/RandomNumberGenerator/RandomNumberGenerator.hpp>

// TODO introduce coldness to model and see how to get it into CSmMALA also
// TODO maybe by introducing it here already!
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
        proposalNegativeLogLikelihood = ModelMixin::calculateNegativeLogLikelihood(MarkovChainProposer::getProposal());
        double acceptanceProbability = stateNegativeLogLikelihood - proposalNegativeLogLikelihood;
        if constexpr(IsCalculateLogAcceptanceProbabilityAvailable<MarkovChainProposer>::value) {
           acceptanceProbability += MarkovChainProposer::calculateLogAcceptanceProbability();
        }
        return acceptanceProbability;
    }

    template<typename MarkovChainProposer, typename ModelImpl>
    double ModelMixin<MarkovChainProposer, ModelImpl>::getNegativeLogLikelihoodOfCurrentState() {
        return stateNegativeLogLikelihood;
    }
}

#endif //HOPS_MODELMIXIN_HPP
