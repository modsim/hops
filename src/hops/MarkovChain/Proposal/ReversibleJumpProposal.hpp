#ifndef HOPS_REVERSIBLEJUMPPROPOSAL_HPP
#define HOPS_REVERSIBLEJUMPPROPOSAL_HPP

#include "Proposal.hpp"
#include "hops/Model/ModelSet.hpp"

namespace hops {
    class ReversibleJumpProposal : public Proposal {
    public:
        VectorType &propose(RandomNumberGenerator &rng) override;

        double computeLogAcceptanceProbability() override;

        VectorType &acceptProposal() override;

        void setState(const VectorType &state) override;

        VectorType getState() const override;

        VectorType getProposal() const override;

        std::vector<std::string> getParameterNames() const override;

        std::any getParameter(const ProposalParameter &parameter) const override;

        std::string getParameterType(const ProposalParameter &parameter) const override;

        void setParameter(const ProposalParameter &parameter, const std::any &value) override;

        bool hasStepSize() const override;

        std::string getProposalName() const override;

        const MatrixType &getA() const override;

        const VectorType &getB() const override;

        std::unique_ptr<Proposal> copyProposal() const override;

    private:
        ModelSet modelSet;

        // The activation vectors also contain elements related to parameters, which are not jumped.
        std::vector<unsigned char> activationProposal;
        std::vector<unsigned char> activationState;

        // StepSize for sampling activated parameter values
        VectorType::Scalar activationStepSize = 0.1;
        std::uniform_real_distribution<double> uniformRealDistribution;
        hops::GaussianStepDistribution<double> gaussianStepDistribution;

        // Fixed value from https://doi.org/10.1093/bioinformatics/btz500
        // TODO add these values to the parameters
        VectorType::Scalar modelJumpProbability = 0.5;
        VectorType::Scalar parameterActivationProbability = 0.1;
        VectorType::Scalar parameterDeactivationProbability = 0.1;

    };

    VectorType &ReversibleJumpProposal::propose(RandomNumberGenerator &rng) {
        return <#initializer#>;
    }

    double ReversibleJumpProposal::computeLogAcceptanceProbability() {
        return 0;
    }

    VectorType &ReversibleJumpProposal::acceptProposal() {
        return <#initializer#>;
    }

    void ReversibleJumpProposal::setState(const VectorType &state) {

    }

    VectorType ReversibleJumpProposal::getState() const {
        return hops::VectorType();
    }

    VectorType ReversibleJumpProposal::getProposal() const {
        return hops::VectorType();
    }

    std::vector<std::string> ReversibleJumpProposal::getParameterNames() const {
        return std::vector<std::string>();
    }

    std::any ReversibleJumpProposal::getParameter(const ProposalParameter &parameter) const {
        return std::any();
    }

    std::string ReversibleJumpProposal::getParameterType(const ProposalParameter &parameter) const {
        return std::string();
    }

    void ReversibleJumpProposal::setParameter(const ProposalParameter &parameter, const std::any &value) {

    }

    bool ReversibleJumpProposal::hasStepSize() const {
        return false;
    }

    std::string ReversibleJumpProposal::getProposalName() const {
        return std::string();
    }

    const MatrixType &ReversibleJumpProposal::getA() const {
        return <#initializer#>;
    }

    const VectorType &ReversibleJumpProposal::getB() const {
        return <#initializer#>;
    }

    std::unique_ptr<Proposal> ReversibleJumpProposal::copyProposal() const {
        return std::unique_ptr<Proposal>();
    }

}

#endif //HOPS_REVERSIBLEJUMPPROPOSAL_HPP
