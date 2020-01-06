#ifndef NUPS_MODEL_HPP
#define NUPS_MODEL_HPP

#include <nups/MarkovChain/Draw/IsCalculateLogAcceptanceProbabilityAvailable.hpp>
#include <nups/RandomNumberGenerator/RandomNumberGenerator.hpp>

// TODO introduce coldness to model and see how to get it into CSmMALA also
// TODO maybe by introducing it here already!
namespace nups {
    /**
     * @brief Model Mixin to add model likelihood to calculateLogAcceptanceRate().
     * @details Useful for MarkovChainProposer classes, that do not already contain the model.
     * @tparam MarkovChainProposer
     * @tparam ModelImpl
     */
    template<typename MarkovChainProposer, typename ModelImpl>
    class Model : public MarkovChainProposer, public ModelImpl {
    public:
        Model(const MarkovChainProposer &markovChainProposer, const ModelImpl &model) :
                MarkovChainProposer(markovChainProposer),
                ModelImpl(model) {
            proposalNegativeLogLikelihood = 0;
            stateNegativeLogLikelihood = ModelImpl::calculateNegativeLogLikelihood(MarkovChainProposer::getState());
        }

                void acceptProposal();

                double calculateLogAcceptanceProbability();
    private:
        double stateNegativeLogLikelihood;
        double proposalNegativeLogLikelihood;
    };

    template<typename MarkovChainProposer, typename ModelImpl>
    void Model<MarkovChainProposer, ModelImpl>::acceptProposal() {
        MarkovChainProposer::acceptProposal();
        stateNegativeLogLikelihood = proposalNegativeLogLikelihood;
    }

    template<typename MarkovChainProposer, typename ModelImpl>
    double Model<MarkovChainProposer, ModelImpl>::calculateLogAcceptanceProbability() {
        proposalNegativeLogLikelihood = Model::calculateNegativeLogLikelihood(MarkovChainProposer::getProposal());
        double acceptanceProbability = stateNegativeLogLikelihood - proposalNegativeLogLikelihood;
        if constexpr(IsCalculateLogAcceptanceProbabilityAvailable<MarkovChainProposer>::value) {
           acceptanceProbability += MarkovChainProposer::calculateLogAcceptanceProbability();
        }
        return acceptanceProbability;
    }
}

#endif //NUPS_MODEL_HPP
