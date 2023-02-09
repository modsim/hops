#ifndef HOPS_REVERSIBLEJUMPPROPOSAL_HPP
#define HOPS_REVERSIBLEJUMPPROPOSAL_HPP

#include <bitset>
#include <Eigen/Core>
#include <memory>
#include <string>
#include <utility>
#include <random>
#include <vector>

#include "hops/Utility/VectorType.hpp"

#include "Proposal.hpp"
#include "ChordStepDistributions.hpp"

namespace hops {
    /**
     * @tparam ProposalImpl is required to have a model mixed in already.
     */
    class ReversibleJumpProposal : public Proposal {
    public:
        ReversibleJumpProposal(std::unique_ptr<Proposal> proposalImpl,
                               const Eigen::VectorXi &jumpIndices,
                               const VectorType &parameterDefaultValues,
                               const std::optional<Eigen::MatrixXd>& A = std::nullopt,
                               const std::optional<Eigen::VectorXd>& b = std::nullopt);

        ReversibleJumpProposal(const ReversibleJumpProposal &other);

        ReversibleJumpProposal(ReversibleJumpProposal &&other) noexcept;

        ReversibleJumpProposal &operator=(const ReversibleJumpProposal &other);

        ReversibleJumpProposal &operator=(ReversibleJumpProposal &&other) noexcept;

        VectorType &propose(RandomNumberGenerator &rng) override;

        VectorType &propose(RandomNumberGenerator &rng, const Eigen::VectorXd &activeIndices) override;

        double computeLogAcceptanceProbability() override;

        VectorType &acceptProposal() override;

        void setState(const VectorType &state) override;

        void setProposal(const VectorType &newProposal) override;

        [[nodiscard]] VectorType getState() const override;

        [[nodiscard]] VectorType getProposal() const override;

        [[nodiscard]] std::optional<double> getStepSize() const override;

        [[nodiscard]] std::vector<std::string> getParameterNames() const override;

        [[nodiscard]] std::any getParameter(const ProposalParameter &parameter) const override;

        [[nodiscard]] std::string getParameterType(const ProposalParameter &parameter) const override;

        [[nodiscard]] std::string getProposalName() const override;

        void setParameter(const ProposalParameter &parameter, const std::any &value) override;

        [[nodiscard]] std::unique_ptr<Proposal> copyProposal() const override;

        void setDimensionNames(const std::vector<std::string> &names) override;

        [[nodiscard]] std::vector<std::string> getDimensionNames() const override;

        [[nodiscard]] const MatrixType &getA() const override;

        [[nodiscard]] const VectorType &getB() const override;

    private:
        VectorType &proposeModel(RandomNumberGenerator &randomNumberGenerator);

        VectorType &wrapProposal(const VectorType &parameterProposal);


        std::unique_ptr<Proposal> proposalImpl;

        // RJMCMC parameters, standard values from https://doi.org/10.1093/bioinformatics/btz500
        VectorType::Scalar modelJumpProbability = 0.5;
        VectorType::Scalar activationProbability = 0.1;
        VectorType::Scalar deactivationProbability = 0.1;

        VectorType backwardDistances;
        VectorType forwardDistances;

        Eigen::VectorXi jumpIndices;
        VectorType defaultValues;
        // VectorType is used instead of VectorXi or some other type because it avoids many casts.
        VectorType activationState;
        VectorType activationProposal;
        VectorType proposal;

        double logAcceptanceChanceModelJump;
        bool lastProposalJumpedModel;

        // Distributions used for proposals, do not have to be copied
        std::uniform_real_distribution<double> uniformRealDistribution;
        hops::UniformStepDistribution<double> stepDistribution;

        Eigen::MatrixXd A;
        Eigen::VectorXd b;
    };
}

#endif //HOPS_REVERSIBLEJUMPPROPOSAL_HPP
