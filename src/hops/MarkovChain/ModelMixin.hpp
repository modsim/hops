#ifndef HOPS_MODELMIXIN_HPP
#define HOPS_MODELMIXIN_HPP

#include <cmath>

#include <hops/MarkovChain/Draw/IsCalculateLogAcceptanceProbabilityAvailable.hpp>
#include <hops/RandomNumberGenerator/RandomNumberGenerator.hpp>
#include <hops/Utility/VectorType.hpp>

namespace hops {
    /**
     * @brief ModelMixin Mixin to add model likelihood to computeLogAcceptanceRate().
     * @details Useful for ProposalType classes, that do not already contain the model.
     * @tparam ProposalType
     * @tparam ModelImpl
     */
    template<typename ProposalType, typename ModelType>
    class ModelMixin : public ProposalType, public ModelType {
    public:
        ModelMixin(const ProposalType &markovChainProposer, const ModelType &model) :
                ProposalType(markovChainProposer),
                ModelType(model) {
            proposalNegativeLogLikelihood = 0;
            stateNegativeLogLikelihood = ModelType::computeNegativeLogLikelihood(ProposalType::getState());
        }

        VectorType &acceptProposal();

        double computeLogAcceptanceProbability();

        double getStateNegativeLogLikelihood();

        double getProposalNegativeLogLikelihood();

        /**
         * @details implementing it here is important to solve the ambivalence, since ModelType and ProposalType both have this function
         * @return
         */
        [[nodiscard]] std::vector<std::string> getDimensionNames() const;

        void setState(const VectorType &);

    private:
        double stateNegativeLogLikelihood;
        double proposalNegativeLogLikelihood;
    };

    template<typename MarkovChainProposer, typename ModelType>
    VectorType &ModelMixin<MarkovChainProposer, ModelType>::acceptProposal() {
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
        return stateNegativeLogLikelihood;
    }

    template<typename MarkovChainProposer, typename ModelType>
    double ModelMixin<MarkovChainProposer, ModelType>::getProposalNegativeLogLikelihood() {
        return proposalNegativeLogLikelihood;
    }

    template<typename MarkovChainProposer, typename ModelType>
    void ModelMixin<MarkovChainProposer, ModelType>::setState(const VectorType &state) {
        MarkovChainProposer::setState(state);
        stateNegativeLogLikelihood = ModelType::computeNegativeLogLikelihood(state);
    }

    template<typename ProposalType, typename ModelType>
    std::vector<std::string> ModelMixin<ProposalType, ModelType>::getDimensionNames() const {
        std::vector<std::string> modelDimensionNames = ModelType::getDimensionNames();
        if (modelDimensionNames.empty()) {
            // If the model does not provide parameter names, return whatever default the proposal returns
            return ProposalType::getDimensionNames();
        }
        return modelDimensionNames;
    }
}

#endif //HOPS_MODELMIXIN_HPP
