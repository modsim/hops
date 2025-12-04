#ifndef HOPS_MODELMIXIN_HPP
#define HOPS_MODELMIXIN_HPP

#include <cmath>

#include "hops/MarkovChain/Draw/IsCalculateLogAcceptanceProbabilityAvailable.hpp"
#include "hops/MarkovChain/Proposal/Proposal.hpp"
#include "hops/Model/Model.hpp"
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

        void resetDistributions() override;

    private:
        ProposalType proposal;
        double coldness = 1.;
        double proposalNegativeLogLikelihood;
        double stateNegativeLogLikelihood;
    };


    template<typename ProposalType>
    class ModelMixin<ProposalType, std::unique_ptr<hops::Model>> : public Proposal, public Model {
    public:
        ModelMixin(const ProposalType &proposal, std::unique_ptr<Model> model) : modelImpl(std::move(model)),
                                                                           proposal(proposal) {
            if (proposal.hasNegativeLogLikelihood()) {
                throw std::invalid_argument("Can't mix in model with ProposalType that already has likelihood.");
            }
            proposalNegativeLogLikelihood = 0;
            stateNegativeLogLikelihood = modelImpl->computeNegativeLogLikelihood(this->proposal.getState());
            std::vector<std::string> modelDimensionNames = modelImpl->getDimensionNames();
            if (!modelDimensionNames.empty()) {
                // If the model does provide dimension names, update the m_proposal dimension names
                this->proposal.setDimensionNames(modelDimensionNames);
            }
        }

        ModelMixin(const ProposalType &proposal, std::unique_ptr<Model> model, double coldness) :
                modelImpl(std::move(model)),
                proposal(proposal),
                coldness(coldness) {
            if (proposal.hasNegativeLogLikelihood()) {
                throw std::invalid_argument("Can't mix in model with ProposalType that already has likelihood.");
            }
            proposalNegativeLogLikelihood = 0;
            stateNegativeLogLikelihood = modelImpl->computeNegativeLogLikelihood(this->proposal.getState());
            std::vector<std::string> modelDimensionNames = modelImpl->getDimensionNames();
            if (!modelDimensionNames.empty()) {
                // If the model does provide dimension names, update the proposal dimension names
                this->proposal.setDimensionNames(modelDimensionNames);
            }
        }

        ModelMixin(const ModelMixin& other) :
			proposal(other.proposal),
            coldness(other.coldness),
            proposalNegativeLogLikelihood(other.proposalNegativeLogLikelihood),
            stateNegativeLogLikelihood(other.stateNegativeLogLikelihood),
            modelImpl(other.modelImpl->copyModel())
        {}

        ModelMixin& operator=(const ModelMixin& other)
        {
            proposal = other.proposal;
            coldness = other.coldness;
            proposalNegativeLogLikelihood = other.proposalNegativeLogLikelihood;
            stateNegativeLogLikelihood   = other.stateNegativeLogLikelihood;
            modelImpl = other.modelImpl->copyModel();
            return *this;
        }

        ModelMixin(ModelMixin&& other) noexcept :
			proposal(std::move(other.proposal)),
            coldness(other.coldness),
            proposalNegativeLogLikelihood(other.proposalNegativeLogLikelihood),
            stateNegativeLogLikelihood(other.stateNegativeLogLikelihood),
            modelImpl(std::move(other.modelImpl))
        {}

        ModelMixin& operator=(ModelMixin&& other) noexcept
        {
            proposal = std::move(other.proposal);
            coldness = other.coldness;
            proposalNegativeLogLikelihood = other.proposalNegativeLogLikelihood;
            stateNegativeLogLikelihood = other.stateNegativeLogLikelihood;
            modelImpl = std::move(other.modelImpl);
            return *this;
        }

        ~ModelMixin() override = default;

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

        void resetDistributions() override;

        typename MatrixType::Scalar computeNegativeLogLikelihood(const VectorType &x) override;

        std::unique_ptr<Model> copyModel() const override;

    private:
        ProposalType proposal;
        double coldness = 1.;
        double proposalNegativeLogLikelihood;
        double stateNegativeLogLikelihood;
        std::unique_ptr<Model> modelImpl;
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

    template<typename ProposalType, typename ModelType>
    void ModelMixin<ProposalType, ModelType>::resetDistributions() {
        proposal.resetDistributions();
    }

    template<typename ProposalType>
    VectorType &ModelMixin<ProposalType, std::unique_ptr<Model>>::propose(RandomNumberGenerator &rng) {
        return proposal.propose(rng);
    }

    template<typename ProposalType>
    VectorType &
    ModelMixin<ProposalType, std::unique_ptr<Model>>::propose(RandomNumberGenerator &rng, const Eigen::VectorXd &activeIndices) {
        return proposal.propose(rng, activeIndices);
    }

    template<typename ProposalType>
    double ModelMixin<ProposalType, std::unique_ptr<Model>>::computeLogAcceptanceProbability() {
        double acceptanceProbability = proposal.computeLogAcceptanceProbability();
        if (std::isfinite(acceptanceProbability)) {
            proposalNegativeLogLikelihood = modelImpl->computeNegativeLogLikelihood(proposal.getProposal());
            acceptanceProbability += coldness * (stateNegativeLogLikelihood - proposalNegativeLogLikelihood);
        }

        return acceptanceProbability;
    }

    template<typename ProposalType>
    VectorType &ModelMixin<ProposalType, std::unique_ptr<Model>>::acceptProposal() {
        stateNegativeLogLikelihood = proposalNegativeLogLikelihood;
        return proposal.acceptProposal();
    }

    template<typename ProposalType>
    void ModelMixin<ProposalType, std::unique_ptr<Model>>::setState(const VectorType &state) {
        proposal.setState(state);
        stateNegativeLogLikelihood = modelImpl->computeNegativeLogLikelihood(state);
    }

    template<typename ProposalType>
    VectorType ModelMixin<ProposalType, std::unique_ptr<Model>>::getState() const {
        return proposal.getState();
    }

    template<typename MarkovChainProposer>
    void ModelMixin<MarkovChainProposer, std::unique_ptr<Model>>::setProposal(const VectorType &proposalVector) {
        proposal.setProposal(proposalVector);
        proposalNegativeLogLikelihood = modelImpl->computeNegativeLogLikelihood(proposalVector);
    }

    template<typename ProposalType>
    VectorType ModelMixin<ProposalType, std::unique_ptr<Model>>::getProposal() const {
        return proposal.getProposal();
    }

    template<typename ProposalType>
    std::vector<std::string> ModelMixin<ProposalType, std::unique_ptr<Model>>::getParameterNames() const {
        std::vector<std::string> parameterNames = {"coldness"};
        std::vector<std::string> proposalParameterNames =  proposal.getParameterNames();
        parameterNames.insert(parameterNames.end(), proposalParameterNames.begin(), proposalParameterNames.end());
        return parameterNames;
    }

    template<typename ProposalType>
    std::any ModelMixin<ProposalType, std::unique_ptr<Model>>::getParameter(const ProposalParameter &parameter) const {
        if (parameter == ProposalParameter::COLDNESS) {
            return std::any(this->coldness);
        }
        return proposal.getParameter(parameter);
    }

    template<typename ProposalType>
    std::string ModelMixin<ProposalType, std::unique_ptr<Model>>::getParameterType(const ProposalParameter &parameter) const {
        if (parameter == ProposalParameter::COLDNESS) {
            return "double";
        }
        return proposal.getParameterType(parameter);
    }

    template<typename ProposalType>
    void ModelMixin<ProposalType, std::unique_ptr<Model>>::setParameter(const ProposalParameter &parameter, const std::any &value) {
        if (parameter == ProposalParameter::COLDNESS) {
            coldness = std::any_cast<double>(value);
        }
        else {
            proposal.setParameter(parameter, value);
        }
    }

    template<typename ProposalType>
    std::string ModelMixin<ProposalType, std::unique_ptr<Model>>::getProposalName() const {
        return proposal.getProposalName() + " + mixed in Model";
    }

    template<typename ProposalType>
    const MatrixType &ModelMixin<ProposalType, std::unique_ptr<Model>>::getA() const {
        return proposal.getA();
    }

    template<typename ProposalType>
    const VectorType &ModelMixin<ProposalType, std::unique_ptr<Model>>::getB() const {
        return proposal.getB();
    }

    template<typename ProposalType>
    std::unique_ptr<Proposal> ModelMixin<ProposalType, std::unique_ptr<Model>>::copyProposal() const {
        return std::make_unique<ModelMixin<ProposalType, std::unique_ptr<Model>>>(*this);
    }

    template<typename ProposalType>
    std::optional<double> ModelMixin<ProposalType, std::unique_ptr<Model>>::getStepSize() const {
        return proposal.getStepSize();
    }

    template<typename ProposalType>
    double ModelMixin<ProposalType, std::unique_ptr<Model>>::getStateNegativeLogLikelihood() {
        return stateNegativeLogLikelihood;
    }

    template<typename ProposalType>
    double ModelMixin<ProposalType, std::unique_ptr<Model>>::getProposalNegativeLogLikelihood() {
        return proposalNegativeLogLikelihood;
    }

    template<typename ProposalType>
    bool ModelMixin<ProposalType, std::unique_ptr<Model>>::hasNegativeLogLikelihood() const {
        return true;
    }

    template<typename ProposalType>
    std::vector<std::string> ModelMixin<ProposalType, std::unique_ptr<Model>>::getDimensionNames() const {
        return proposal.getDimensionNames();
    }

    template<typename ProposalType>
    bool ModelMixin<ProposalType, std::unique_ptr<Model>>::isSymmetric() const {
        return proposal.isSymmetric();
    }

    template<typename ProposalType>
    void ModelMixin<ProposalType, std::unique_ptr<Model>>::setDimensionNames(const std::vector<std::string> &names) {
        proposal.setDimensionNames(names);
    }

    template<typename ProposalType>
    void ModelMixin<ProposalType, std::unique_ptr<Model>>::resetDistributions() {
        proposal.resetDistributions();
    }

    template<typename ProposalType>
    typename MatrixType::Scalar ModelMixin<ProposalType, std::unique_ptr<Model>>::computeNegativeLogLikelihood(const VectorType &x) {
		return modelImpl->computeNegativeLogLikelihood(x);
    }

    template<typename ProposalType>
    std::unique_ptr<Model> ModelMixin<ProposalType, std::unique_ptr<Model>>::copyModel() const {
		return modelImpl->copyModel();
    }

}// namespace hops

#endif//HOPS_MODELMIXIN_HPP
