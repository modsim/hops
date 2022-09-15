#ifndef HOPS_REVERSIBLEJUMPPROPOSAL_HPP
#define HOPS_REVERSIBLEJUMPPROPOSAL_HPP

#include <bitset>
#include <Eigen/Core>
#include <string>
#include <utility>
#include <vector>
#include <random>

#include <hops/MarkovChain/Proposal/ChordStepDistributions.hpp>
#include <hops/MarkovChain/Proposal/Proposal.hpp>
#include <hops/RandomNumberGenerator/RandomNumberGenerator.hpp>
#include <hops/Utility/VectorType.hpp>
#include <hops/Utility/StringUtility.hpp>

#include "Proposal.hpp"

namespace {
    std::pair<double, double> distanceInCoordinateDirection(const Eigen::MatrixXd &A,
                                                            const Eigen::VectorXd &b,
                                                            double currentValue,
                                                            long coordinate) {
        Eigen::VectorXd slacks = b - A.col(coordinate) * currentValue;
        Eigen::VectorXd inverseDistances = A.col(coordinate).cwiseQuotient(slacks);
        // Inverse distance are potentially nan due to default values on the boundary of the polytope.
        // Replaces nan because nan should not influence the distances.
        inverseDistances = inverseDistances
                .array()
                .unaryExpr([](double value) { return std::isnan(value) ? 0. : value; })
                .matrix();
        double forwardDistance = 1. / inverseDistances.maxCoeff();
        double backwardDistance = 1. / inverseDistances.minCoeff();
        forwardDistance = (forwardDistance < 0) ? 0 : forwardDistance;
        backwardDistance = (backwardDistance > 0) ? 0 : backwardDistance;
        assert(backwardDistance <= 0 && forwardDistance >= 0);
        return std::make_pair(backwardDistance, forwardDistance);
    }
}

namespace hops {
    /**
     * @tparam ProposalImpl is required to have a model mixed in already.
     */
    class ReversibleJumpProposal : public Proposal {
    public:
        ReversibleJumpProposal(std::unique_ptr<Proposal> proposalImpl,
                               const Eigen::VectorXi &jumpIndices,
                               const VectorType &parameterDefaultValues);

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

        VectorType defaultValues;
        Eigen::VectorXi jumpIndices;
        // VectorType is used instead of VectorXi or some other type because it avoids many casts.
        VectorType activationState;
        VectorType activationProposal;
        VectorType proposal;

        double logAcceptanceChanceModelJump;
        bool lastProposalJumpedModel;

        // Distributions used for proposals, do not have to be copied
        std::uniform_real_distribution<double> uniformRealDistribution;
        hops::UniformStepDistribution<double> stepDistribution;
    };

    ReversibleJumpProposal::ReversibleJumpProposal(std::unique_ptr<Proposal> proposalImpl,
                                                   const Eigen::VectorXi &jumpIndices,
                                                   const VectorType &parameterDefaultValues) :
            proposalImpl(std::move(proposalImpl)),
            jumpIndices(jumpIndices),
            defaultValues(parameterDefaultValues) {

        if (this->jumpIndices.rows() != this->defaultValues.rows()) {
            throw std::runtime_error("dimension missmatch in input");
        }

        VectorType parameterState = this->proposalImpl->getState();
        this->activationState = Eigen::VectorXd::Ones(parameterState.rows());
        // precomputes backward & forwards distances. Works because we use uniform jumping distribution.
        this->backwardDistances = VectorType::Zero(this->jumpIndices.rows());
        this->forwardDistances = VectorType::Zero(this->jumpIndices.rows());
        for (long i = 0; i < this->jumpIndices.rows(); ++i) {
            parameterState(this->jumpIndices(i)) = this->defaultValues(i);
            // Starts with all optional parameters deactivated, which is the simplest model
            this->activationState(jumpIndices(i)) = 0.;
            auto[b, f] = distanceInCoordinateDirection(this->proposalImpl->getA(),
                                                       this->proposalImpl->getB(),
                                                       this->defaultValues(i),
                                                       this->jumpIndices(i));
            this->backwardDistances(i) = b;
            this->forwardDistances(i) = f;
        }
        this->activationProposal = activationState;

        this->proposalImpl->setState(parameterState);
        this->logAcceptanceChanceModelJump = 0;
        this->proposal = VectorType::Zero(activationProposal.rows() + parameterState.rows());
        this->lastProposalJumpedModel = false;
    }

    ReversibleJumpProposal::ReversibleJumpProposal(const ReversibleJumpProposal &other) {
        this->proposalImpl = other.proposalImpl->copyProposal();
        this->jumpIndices = other.jumpIndices;
        this->defaultValues = other.defaultValues;
        this->activationState = other.activationState;
        this->activationProposal = other.activationProposal;
        this->proposal = other.proposal;
        this->lastProposalJumpedModel = other.lastProposalJumpedModel;
        this->logAcceptanceChanceModelJump = other.logAcceptanceChanceModelJump;
        this->backwardDistances = other.backwardDistances;
        this->forwardDistances = other.forwardDistances;
    }

    ReversibleJumpProposal::ReversibleJumpProposal(ReversibleJumpProposal &&other) noexcept {
        this->proposalImpl = std::move(other.proposalImpl);
        this->jumpIndices = other.jumpIndices;
        this->defaultValues = other.defaultValues;
        this->activationState = other.activationState;
        this->activationProposal = other.activationProposal;
        this->proposal = other.proposal;
        this->lastProposalJumpedModel = other.lastProposalJumpedModel;
        this->logAcceptanceChanceModelJump = other.logAcceptanceChanceModelJump;
        this->backwardDistances = other.backwardDistances;
        this->forwardDistances = other.forwardDistances;
    }

    ReversibleJumpProposal &ReversibleJumpProposal::operator=(const ReversibleJumpProposal &other) {
        this->proposalImpl = other.proposalImpl->copyProposal();
        this->jumpIndices = other.jumpIndices;
        this->defaultValues = other.defaultValues;
        this->activationState = other.activationState;
        this->activationProposal = other.activationProposal;
        this->proposal = other.proposal;
        this->lastProposalJumpedModel = other.lastProposalJumpedModel;
        this->logAcceptanceChanceModelJump = other.logAcceptanceChanceModelJump;
        this->backwardDistances = other.backwardDistances;
        this->forwardDistances = other.forwardDistances;
        return *this;
    }

    ReversibleJumpProposal &ReversibleJumpProposal::operator=(ReversibleJumpProposal &&other) noexcept {
        this->proposalImpl = std::move(other.proposalImpl);
        this->jumpIndices = other.jumpIndices;
        this->defaultValues = other.defaultValues;
        this->activationState = other.activationState;
        this->activationProposal = other.activationProposal;
        this->proposal = other.proposal;
        this->lastProposalJumpedModel = other.lastProposalJumpedModel;
        this->logAcceptanceChanceModelJump = other.logAcceptanceChanceModelJump;
        this->backwardDistances = other.backwardDistances;
        this->forwardDistances = other.forwardDistances;
        return *this;
    }

    VectorType &ReversibleJumpProposal::propose(RandomNumberGenerator &rng) {
        if (uniformRealDistribution(rng) < modelJumpProbability) {
            lastProposalJumpedModel = true;
            proposal = proposeModel(rng);
        } else {
            lastProposalJumpedModel = false;
            this->activationProposal = this->activationState;
            proposal = wrapProposal(proposalImpl->propose(rng, activationState));
        }
        return proposal;
    }

    double ReversibleJumpProposal::computeLogAcceptanceProbability() {
        if (lastProposalJumpedModel) {
            return this->logAcceptanceChanceModelJump;
        }
        return proposalImpl->computeLogAcceptanceProbability();
    }

    VectorType &ReversibleJumpProposal::acceptProposal() {
        activationState = activationProposal;
        return wrapProposal(proposalImpl->acceptProposal());
    }

    void ReversibleJumpProposal::setState(const VectorType &state) {
        this->activationState = state.topRows(activationState.rows());
        proposalImpl->setState(state.bottomRows(state.rows() - this->activationState.rows()));
    }

    void ReversibleJumpProposal::setProposal(const VectorType &newProposal) {
        proposal = newProposal;
        activationProposal = newProposal.topRows(this->activationProposal.rows());
        proposalImpl->setProposal(proposal.bottomRows(proposal.rows() - this->activationProposal.rows()));
    }

    VectorType ReversibleJumpProposal::getState() const {
        VectorType parameterState = proposalImpl->getState();
        VectorType state(activationState.rows() + parameterState.rows());
        state << activationState, parameterState;
        return state;
    }

    std::vector<std::string> ReversibleJumpProposal::getDimensionNames() const {
        // Vector is constructed on demand, because it typically is not used repeatedly.
        std::vector<std::string> dimensionNames = proposalImpl->getDimensionNames();
        std::vector<std::string> names;
        for (long i = 0; i < jumpIndices.rows(); ++i) {
            names.emplace_back(dimensionNames[i] + "_activation");
        }
        names.insert(names.end(), dimensionNames.begin(), dimensionNames.end());
        return names;
    }

    VectorType ReversibleJumpProposal::getProposal() const {
        return proposal;
    }

    std::vector<std::string> ReversibleJumpProposal::getParameterNames() const {
        std::vector<std::string> parameterNames = proposalImpl->getParameterNames();
        parameterNames.emplace_back(
                ProposalParameterName[static_cast<int>(ProposalParameter::MODEL_JUMP_PROBABILITY)]);
        parameterNames.emplace_back(
                ProposalParameterName[static_cast<int>(ProposalParameter::ACTIVATION_PROBABILITY)]);
        parameterNames.emplace_back(
                ProposalParameterName[static_cast<int>(ProposalParameter::DEACTIVATION_PROBABILITY)]);
        return parameterNames;
    }

    std::any ReversibleJumpProposal::getParameter(const ProposalParameter &parameter) const {
        switch (parameter) {
            case ProposalParameter::MODEL_JUMP_PROBABILITY:
                return modelJumpProbability;
            case ProposalParameter::ACTIVATION_PROBABILITY:
                return activationProbability;
            case ProposalParameter::DEACTIVATION_PROBABILITY:
                return deactivationProbability;
            default:
                return proposalImpl->getParameter(parameter);
        }
    }

    std::string ReversibleJumpProposal::getParameterType(const ProposalParameter &parameter) const {
        if (parameter == ProposalParameter::MODEL_JUMP_PROBABILITY ||
            parameter == ProposalParameter::ACTIVATION_PROBABILITY ||
            parameter == ProposalParameter::DEACTIVATION_PROBABILITY) {
            return "double";
        } else {
            return proposalImpl->getParameterType(parameter);
        }
    }

    void ReversibleJumpProposal::setParameter(const ProposalParameter &parameter, const std::any &value) {
        switch (parameter) {
            case ProposalParameter::MODEL_JUMP_PROBABILITY:
                if (std::any_cast<double>(value) >= 1.) {
                    throw std::invalid_argument(std::string(
                            ProposalParameterName[static_cast<int>(ProposalParameter::MODEL_JUMP_PROBABILITY)]) +
                                                " can not be set to 1 or larger.");
                }
                modelJumpProbability = std::any_cast<double>(value);
                break;
            case ProposalParameter::ACTIVATION_PROBABILITY:
                if (std::any_cast<double>(value) >= 1.) {
                    throw std::invalid_argument(std::string(
                            ProposalParameterName[static_cast<int>(ProposalParameter::ACTIVATION_PROBABILITY)]) +
                                                " can not be set to 1 or larger.");
                }
                activationProbability = std::any_cast<double>(value);
                break;
            case ProposalParameter::DEACTIVATION_PROBABILITY:
                if (std::any_cast<double>(value) >= 1.) {
                    throw std::invalid_argument(std::string(
                            ProposalParameterName[static_cast<int>(ProposalParameter::DEACTIVATION_PROBABILITY)]) +
                                                " can not be set to 1 or larger.");
                }
                deactivationProbability = std::any_cast<double>(value);
                break;
            default:
                proposalImpl->setParameter(parameter, value);
        }
    }

    std::string ReversibleJumpProposal::getProposalName() const {
        return "RJMCMC(" + proposalImpl->getProposalName() + ")";
    }

    std::unique_ptr<Proposal> ReversibleJumpProposal::copyProposal() const {
        return std::make_unique<ReversibleJumpProposal>(*this);
    }

    const MatrixType &ReversibleJumpProposal::getA() const {
        return proposalImpl->getA();
    }

    const VectorType &ReversibleJumpProposal::getB() const {
        return proposalImpl->getB();
    }

    VectorType &ReversibleJumpProposal::proposeModel(RandomNumberGenerator &randomNumberGenerator) {
        VectorType parameterProposal = proposalImpl->getState();
        this->activationProposal = this->activationState;
        this->logAcceptanceChanceModelJump = 0;

        for (long i = 0; i < jumpIndices.rows(); ++i) {
            long jumpIndex = this->jumpIndices(i);
            bool isActive = this->activationState(jumpIndex) != 0;
            auto jumpProbability = isActive ? this->deactivationProbability
                                            : this->activationProbability;

            if (uniformRealDistribution(randomNumberGenerator) < jumpProbability) {
                this->activationProposal(jumpIndex) = !isActive;
            } else {
                this->activationProposal(jumpIndex) = isActive;
            }

            if (!this->activationProposal(jumpIndex) && this->activationState(jumpIndex)) {
                // deactivate
                logAcceptanceChanceModelJump += std::log(activationProbability);
                logAcceptanceChanceModelJump -= std::log(deactivationProbability);
                parameterProposal(jumpIndex) = defaultValues(i);
            } else if (this->activationProposal(jumpIndex) && !this->activationState(jumpIndex)) {
                // activate
                logAcceptanceChanceModelJump -= std::log(activationProbability);
                logAcceptanceChanceModelJump += std::log(deactivationProbability);
                parameterProposal(jumpIndex) = defaultValues(i) + stepDistribution.draw(randomNumberGenerator,
                                                                                        backwardDistances(i),
                                                                                        forwardDistances(i));
            }
        }

        proposalImpl->setProposal(parameterProposal);
        this->logAcceptanceChanceModelJump += proposalImpl->getStateNegativeLogLikelihood()
                                              - proposalImpl->getProposalNegativeLogLikelihood();

        return wrapProposal(parameterProposal);
    }

    std::optional<double> ReversibleJumpProposal::getStepSize() const {
        return this->proposalImpl->getStepSize();
    }

    VectorType &ReversibleJumpProposal::propose(RandomNumberGenerator &rng, const Eigen::VectorXd &activeIndices) {
        return wrapProposal(proposalImpl->propose(rng, activeIndices));
    }

    VectorType &ReversibleJumpProposal::wrapProposal(const VectorType &parameterProposal) {
        proposal.setZero();
        proposal << this->activationProposal, parameterProposal;
        return proposal;
    }
}

#endif //HOPS_REVERSIBLEJUMPPROPOSAL_HPP
