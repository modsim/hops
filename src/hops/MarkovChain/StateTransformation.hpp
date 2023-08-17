#ifndef HOPS_STATETRANSFORMATION_HPP
#define HOPS_STATETRANSFORMATION_HPP

#include <vector>
#include <any>
#include <string>
#include <memory>
#include <optional>
#include "Proposal/Proposal.hpp"
#include "hops/Utility/VectorType.hpp"

namespace hops {
    /**
     * @brief Mixin for undoing transformations to the Markov chain state.
     * @details Prominent use-case is for dealing with rounding.
     * @tparam ProposalImpl
     */
    template<typename ProposalImpl, typename TransformationImpl>
    class StateTransformation : public Proposal {
    public:
        explicit StateTransformation(const ProposalImpl &proposalImpl, const TransformationImpl &transformation) :
                proposalImpl(proposalImpl),
                transformation(transformation) {}

        [[nodiscard]] VectorType getState() const override {
            return transformation.apply(proposalImpl.getState());
        }

        [[nodiscard]] VectorType getProposal() const override {
            return transformation.apply(proposalImpl.getProposal());
        }

        VectorType &propose(RandomNumberGenerator &rng) override {
            stateStorage = proposalImpl.propose(rng);
            stateStorage = transformation.apply(stateStorage);
            return stateStorage;
        }

        VectorType &propose(RandomNumberGenerator &rng, const Eigen::VectorXd &activeSubspaces) override {
            stateStorage = proposalImpl.propose(rng, activeSubspaces);
            stateStorage = transformation.apply(stateStorage);
            return stateStorage;
        }

        VectorType &acceptProposal() override {
            stateStorage = proposalImpl.acceptProposal();
            stateStorage = transformation.apply(stateStorage);
            return stateStorage;
        }

        void setState(const VectorType &state) override {
            proposalImpl.setState(transformation.revert(state));
        }

        void setProposal(const VectorType &proposal) override {
            proposalImpl.setProposal(transformation.revert(proposal));
        }

        double computeLogAcceptanceProbability() override {
            return proposalImpl.computeLogAcceptanceProbability();
        }

        void setDimensionNames(const std::vector<std::string> &names) override {
            proposalImpl.setDimensionNames(names);
        }

        std::vector<std::string> getDimensionNames() const override {
            return proposalImpl.getDimensionNames();
        }

        std::vector<std::string> getParameterNames() const override {
            return proposalImpl.getParameterNames();
        }

        std::any getParameter(const ProposalParameter &parameter) const override {
            return proposalImpl.getParameter(parameter);
        }

        std::string getParameterType(const ProposalParameter &parameter) const override {
            return proposalImpl.getParameterType(parameter);
        }

        void setParameter(const ProposalParameter &parameter, const std::any &value) override {
            proposalImpl.setParameter(parameter, value);
        }

        std::string getProposalName() const override {
            return proposalImpl.getProposalName();
        }

        const MatrixType &getA() const override {
            return proposalImpl.getA();
        }

        const VectorType &getB() const override {
            return proposalImpl.getB();
        }

        std::unique_ptr<Proposal> copyProposal() const override {
            return std::make_unique<StateTransformation<ProposalImpl, TransformationImpl>>(*this);
        }

        std::optional<double> getStepSize() const override {
            return proposalImpl.getStepSize();
        }

        double getStateNegativeLogLikelihood() override {
            return proposalImpl.getStateNegativeLogLikelihood();
        }

        double getProposalNegativeLogLikelihood() override {
            return proposalImpl.getProposalNegativeLogLikelihood();
        }

        bool hasNegativeLogLikelihood() const override {
            return proposalImpl.hasNegativeLogLikelihood();
        }

        bool isSymmetric() const override {
            return proposalImpl.isSymmetric();
        }

    private:
        ProposalImpl proposalImpl;
        TransformationImpl transformation;
        VectorType stateStorage; // used for cases where non-const lvalues could otherwise not bind.
    };
}

#endif //HOPS_STATETRANSFORMATION_HPP
