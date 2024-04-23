#ifndef HOPS_MODELMIXIN_HPP
#define HOPS_MODELMIXIN_HPP

#include <cmath>

#include "hops/MarkovChain/Draw/IsCalculateLogAcceptanceProbabilityAvailable.hpp"
#include "hops/MarkovChain/Proposal/Proposal.hpp"
#include "hops/RandomNumberGenerator/RandomNumberGenerator.hpp"
#include "hops/Utility/VectorType.hpp"

namespace hops {
    /**
     * @brief ModelMixin Mixin to add model likelihood to computeLogAcceptanceRate().
     * @details Useful for ProposalType classes, that do not already contain the model.
     * @tparam ProposalType
     * @tparam ModelImpl
     */
    template<typename ProposalType, typename ModelType>
    class ModelMixin : public Proposal, public ModelType {
    public:
        ModelMixin(const ProposalType &proposal, const ModelType &model) : ModelType(model),
                                                                           proposal(proposal) {
            if (proposal.hasNegativeLogLikelihood()) {
                throw std::invalid_argument("Can't mix in model with ProposalType that already has likelihood.");
            }
            proposalNegativeLogLikelihood = 0;
            stateNegativeLogLikelihood = ModelType::computeNegativeLogLikelihood(this->proposal.getState());
            std::vector<std::string> modelDimensionNames = ModelType::getDimensionNames();
            if (!modelDimensionNames.empty()) {
                // If the model does provide dimension names, update the m_proposal dimension names
                this->proposal.setDimensionNames(modelDimensionNames);
            }
        }

        ModelMixin(const ProposalType &proposal, const ModelType &model, double coldness) :
                ModelType(model),
                proposal(proposal),
                coldness(coldness) {
            if (proposal.hasNegativeLogLikelihood()) {
                throw std::invalid_argument("Can't mix in model with ProposalType that already has likelihood.");
            }
            proposalNegativeLogLikelihood = 0;
            stateNegativeLogLikelihood = ModelType::computeNegativeLogLikelihood(this->proposal.getState());
            std::vector<std::string> modelDimensionNames = ModelType::getDimensionNames();
            if (!modelDimensionNames.empty()) {
                // If the model does provide dimension names, update the proposal dimension names
                this->proposal.setDimensionNames(modelDimensionNames);
            }
        }

        VectorType &propose(RandomNumberGenerator &rng) override;

        VectorType &propose(RandomNumberGenerator &rng, const Eigen::VectorXd &activeIndices) override;

        double computeLogAcceptanceProbability() override;

        VectorType &acceptProposal() override;

        void setState(const VectorType &state) override;

        void setProposal(const VectorType &proposal) override;

        [[nodiscard]] VectorType getState() const override;

        [[nodiscard]] VectorType getProposal() const override;

        [[nodiscard]] std::optional<double> getStepSize() const override;

        [[nodiscard]] double getStateNegativeLogLikelihood() override;

        [[nodiscard]] double getProposalNegativeLogLikelihood() override;

        [[nodiscard]] bool hasNegativeLogLikelihood() const override;

        [[nodiscard]] std::vector<std::string> getParameterNames() const override;

        [[nodiscard]] std::any getParameter(const ProposalParameter &parameter) const override;

        [[nodiscard]] std::string getParameterType(const ProposalParameter &parameter) const override;

        void setParameter(const ProposalParameter &parameter, const std::any &value) override;

        [[nodiscard]] std::string getProposalName() const override;

        [[nodiscard]] const MatrixType &getA() const override;

        [[nodiscard]] const VectorType &getB() const override;

        void setDimensionNames(const std::vector<std::string> &names) override;

        [[nodiscard]] std::vector<std::string> getDimensionNames() const override;

        [[nodiscard]] std::unique_ptr<Proposal> copyProposal() const override;

        [[nodiscard]] bool isSymmetric() const override;

    private:
        ProposalType proposal;
        double coldness = 1.;
        double proposalNegativeLogLikelihood;
        double stateNegativeLogLikelihood;
    };

    template<typename ProposalType, typename ModelType>
    VectorType &ModelMixin<ProposalType, ModelType>::propose(RandomNumberGenerator &rng) {
        return proposal.propose(rng);
    }

    template<typename ProposalType, typename ModelType>
    VectorType &
    ModelMixin<ProposalType, ModelType>::propose(RandomNumberGenerator &rng, const Eigen::VectorXd &activeIndices) {
        return proposal.propose(rng, activeIndices);
    }

    template<typename ProposalType, typename ModelType>
    double ModelMixin<ProposalType, ModelType>::computeLogAcceptanceProbability() {
        double acceptanceProbability = proposal.computeLogAcceptanceProbability();

        if (std::isfinite(acceptanceProbability)) {
            proposalNegativeLogLikelihood = ModelType::computeNegativeLogLikelihood(proposal.getProposal());
            acceptanceProbability += coldness * (stateNegativeLogLikelihood - proposalNegativeLogLikelihood);
        }

        return acceptanceProbability;
    }

    template<typename ProposalType, typename ModelType>
    VectorType &ModelMixin<ProposalType, ModelType>::acceptProposal() {
        stateNegativeLogLikelihood = proposalNegativeLogLikelihood;
        return proposal.acceptProposal();
    }

    template<typename ProposalType, typename ModelType>
    void ModelMixin<ProposalType, ModelType>::setState(const VectorType &state) {
        proposal.setState(state);
        stateNegativeLogLikelihood = ModelType::computeNegativeLogLikelihood(state);
    }

    template<typename ProposalType, typename ModelType>
    VectorType ModelMixin<ProposalType, ModelType>::getState() const {
        return proposal.getState();
    }

    template<typename MarkovChainProposer, typename ModelType>
    void ModelMixin<MarkovChainProposer, ModelType>::setProposal(const VectorType &proposalVector) {
        proposal.setProposal(proposalVector);
        proposalNegativeLogLikelihood = ModelType::computeNegativeLogLikelihood(proposalVector);
    }

    template<typename ProposalType, typename ModelType>
    VectorType ModelMixin<ProposalType, ModelType>::getProposal() const {
        return proposal.getProposal();
    }

    template<typename ProposalType, typename ModelType>
    std::vector<std::string> ModelMixin<ProposalType, ModelType>::getParameterNames() const {
        std::vector<std::string> parameterNames = {"coldness"};
        std::vector<std::string> proposalParameterNames =  proposal.getParameterNames();
        parameterNames.insert(parameterNames.end(), proposalParameterNames.begin(), proposalParameterNames.end());
        return parameterNames;
    }

    template<typename ProposalType, typename ModelType>
    std::any ModelMixin<ProposalType, ModelType>::getParameter(const ProposalParameter &parameter) const {
        if (parameter == ProposalParameter::COLDNESS) {
            return std::any(this->coldness);
        }
        return proposal.getParameter(parameter);
    }

    template<typename ProposalType, typename ModelType>
    std::string ModelMixin<ProposalType, ModelType>::getParameterType(const ProposalParameter &parameter) const {
        if (parameter == ProposalParameter::COLDNESS) {
            return "double";
        }
        return proposal.getParameterType(parameter);
    }

    template<typename ProposalType, typename ModelType>
    void ModelMixin<ProposalType, ModelType>::setParameter(const ProposalParameter &parameter, const std::any &value) {
        if (parameter == ProposalParameter::COLDNESS) {
            coldness = std::any_cast<double>(value);
        }
        else {
            proposal.setParameter(parameter, value);
        }
    }

    template<typename ProposalType, typename ModelType>
    std::string ModelMixin<ProposalType, ModelType>::getProposalName() const {
        return proposal.getProposalName() + " + mixed in Model";
    }

    template<typename ProposalType, typename ModelType>
    const MatrixType &ModelMixin<ProposalType, ModelType>::getA() const {
        return proposal.getA();
    }

    template<typename ProposalType, typename ModelType>
    const VectorType &ModelMixin<ProposalType, ModelType>::getB() const {
        return proposal.getB();
    }

    template<typename ProposalType, typename ModelType>
    std::unique_ptr<Proposal> ModelMixin<ProposalType, ModelType>::copyProposal() const {
        return std::make_unique<ModelMixin<ProposalType, ModelType>>(*this);
    }

    template<typename ProposalType, typename ModelType>
    std::optional<double> ModelMixin<ProposalType, ModelType>::getStepSize() const {
        return proposal.getStepSize();
    }

    template<typename ProposalType, typename ModelType>
    double ModelMixin<ProposalType, ModelType>::getStateNegativeLogLikelihood() {
        return stateNegativeLogLikelihood;
    }

    template<typename ProposalType, typename ModelType>
    double ModelMixin<ProposalType, ModelType>::getProposalNegativeLogLikelihood() {
        return proposalNegativeLogLikelihood;
    }

    template<typename ProposalType, typename ModelType>
    bool ModelMixin<ProposalType, ModelType>::hasNegativeLogLikelihood() const {
        return true;
    }

    template<typename ProposalType, typename ModelType>
    std::vector<std::string> ModelMixin<ProposalType, ModelType>::getDimensionNames() const {
        return proposal.getDimensionNames();
    }

    template<typename ProposalType, typename ModelType>
    bool ModelMixin<ProposalType, ModelType>::isSymmetric() const {
        return proposal.isSymmetric();
    }

    template<typename ProposalType, typename ModelType>
    void ModelMixin<ProposalType, ModelType>::setDimensionNames(const std::vector<std::string> &names) {
        proposal.setDimensionNames(names);
    }
}// namespace hops

#endif//HOPS_MODELMIXIN_HPP
