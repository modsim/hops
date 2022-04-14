#ifndef HOPS_TRUNCATEDGAUSSIANPROPOSAL_HPP
#define HOPS_TRUNCATEDGAUSSIANPROPOSAL_HPP

#include <random>

#include <hops/Model/Gaussian.hpp>
#include <hops/RandomNumberGenerator/RandomNumberGenerator.hpp>
#include <hops/Utility/MatrixType.hpp>
#include <hops/Utility/StringUtility.hpp>
#include <hops/Utility/VectorType.hpp>

#include "ChordStepDistributions.hpp"
#include "IsSetStepSizeAvailable.hpp"
#include "Proposal.hpp"

namespace hops {
    template<typename InternalMatrixType, typename InternalVectorType>
    class TruncatedGaussianProposal : public Proposal, public Gaussian {
    public:

        /**
         * @brief Constructs Coordinate Hit and Run proposal mechanism on polytope defined as Ax<b.
         * @param A
         * @param b
         * @param currentState
         */
        TruncatedGaussianProposal(InternalMatrixType A,
                                  InternalVectorType b,
                                  InternalVectorType currentState,
                                  const Gaussian &model);

        VectorType &propose(RandomNumberGenerator &rng) override;

        VectorType &acceptProposal() override;

        void setState(const VectorType &state) override;

        [[nodiscard]] VectorType getState() const override;

        [[nodiscard]] VectorType getProposal() const override;

        [[nodiscard]] std::vector<std::string> getParameterNames() const override;

        [[nodiscard]] std::any getParameter(const ProposalParameter &parameter) const override;

        [[nodiscard]] std::string getParameterType(const ProposalParameter &parameter) const override;

        void setParameter(const ProposalParameter &parameter, const std::any &value) override;

        [[nodiscard]] std::optional<double> getStepSize() const;

        [[nodiscard]] bool hasStepSize() const override;

        [[nodiscard]] std::string getProposalName() const override;

        [[nodiscard]] std::unique_ptr<Proposal> copyProposal() const override;

        [[nodiscard]] double computeLogAcceptanceProbability() override;

        [[nodiscard]] const MatrixType &getA() const override;

        [[nodiscard]] const VectorType &getB() const override;

        ProposalStatistics &getProposalStatistics() override;

        void activateTrackingOfProposalStatistics() override;

        void disableTrackingOfProposalStatistics() override;

        bool isTrackingOfProposalStatisticsActivated() override;

        [[nodiscard]] std::unique_ptr<Model> getModel() const;

        ProposalStatistics getAndResetProposalStatistics() override;

        std::vector<std::string> getDimensionNames() const override;

    private:
        InternalMatrixType A;
        InternalVectorType b;

        VectorType state;
        VectorType proposal;
        InternalVectorType slacks;
        InternalVectorType inverseDistances;
        ProposalStatistics proposalStatistics;

        typename InternalMatrixType::Scalar step = 0;
        GaussianStepDistribution<VectorType::Scalar> chordStepDistribution;
        typename InternalMatrixType::Scalar forwardDistance = 0;
        typename InternalMatrixType::Scalar backwardDistance = 0;

        bool isProposalInfosTrackingActive = false;
    };

    template<typename InternalMatrixType, typename InternalVectorType>
    TruncatedGaussianProposal<InternalMatrixType, InternalVectorType>::TruncatedGaussianProposal(
            InternalMatrixType A_,
            InternalVectorType b_,
            InternalVectorType currentState_,
            const Gaussian &model) :
            Gaussian(model),
            A(std::move(A_)),
            b(std::move(b_)),
            state(std::move(currentState_)),
            proposal(this->state) {
        if (((b - A * state).array() < 0).any()) {
            throw std::invalid_argument("Starting point outside polytope always gives constant Markov chain.");
        }

        slacks = this->b - this->A * this->state;
    }

    template<typename InternalMatrixType, typename InternalVectorType>
    VectorType &TruncatedGaussianProposal<InternalMatrixType, InternalVectorType>::propose(
            RandomNumberGenerator &rng) {
        MatrixType cholesky = Gaussian::getCovarianceLowerCholesky();
        const VectorType &mean = Gaussian::getMean();
        VectorType shiftedState = state - mean;
        VectorType whiteState = cholesky.template triangularView<Eigen::Lower>().
                template solve(shiftedState);
        proposal = whiteState;
        MatrixType whitenedA = A * cholesky.template triangularView<Eigen::Lower>();
        VectorType whitenedB = b - A * mean;

        for (long i = 0; i < state.rows(); ++i) {
            slacks = whitenedB - whitenedA * proposal;
            inverseDistances = whitenedA.col(i).cwiseQuotient(slacks);
            forwardDistance = 1. / inverseDistances.maxCoeff();
            backwardDistance = 1. / inverseDistances.minCoeff();
            if (forwardDistance < 0) {
                forwardDistance = std::numeric_limits<typename InternalMatrixType::Scalar>::infinity();
            }
            if (backwardDistance > 0) {
                backwardDistance = -std::numeric_limits<typename InternalMatrixType::Scalar>::infinity();
            }

            step = chordStepDistribution.draw(rng, 1., backwardDistance, forwardDistance);

            proposal(i) += step;
        }

        proposal = cholesky * (proposal) + mean;
        // A * (L * (wS) + mu) < b
        // A * L * ws < b - A * mu
        return proposal;
    }

    template<typename InternalMatrixType, typename InternalVectorType>
    VectorType &
    TruncatedGaussianProposal<InternalMatrixType, InternalVectorType>::acceptProposal() {
        state.swap(proposal);
        return state;
    }

    template<typename InternalMatrixType, typename InternalVectorType>
    void TruncatedGaussianProposal<InternalMatrixType, InternalVectorType>::setState(
            const VectorType &newState) {
        if (((b - A * newState).array() < 0).any()) {
            throw std::invalid_argument("Starting point outside polytope always gives constant Markov chain.");
        }
        TruncatedGaussianProposal::state = newState;
        TruncatedGaussianProposal::proposal = TruncatedGaussianProposal::state;
        slacks = b - A * TruncatedGaussianProposal::state;
    }

    template<typename InternalMatrixType, typename InternalVectorType>
    std::optional<double>
    TruncatedGaussianProposal<InternalMatrixType, InternalVectorType>::getStepSize() const {
        return std::nullopt;
    }

    template<typename InternalMatrixType, typename InternalVectorType>
    std::string
    TruncatedGaussianProposal<InternalMatrixType, InternalVectorType>::getProposalName() const {
        return "TruncatedGaussianProposal";
    }

    template<typename InternalMatrixType, typename InternalVectorType>
    double
    TruncatedGaussianProposal<InternalMatrixType, InternalVectorType>::computeLogAcceptanceProbability() {
        return Gaussian::computeNegativeLogLikelihood(proposal)
               - Gaussian::computeNegativeLogLikelihood(state);
    }

    template<typename InternalMatrixType, typename InternalVectorType>
    bool
    TruncatedGaussianProposal<InternalMatrixType, InternalVectorType>::hasStepSize() const {
        return false;
    }

    template<typename InternalMatrixType, typename InternalVectorType>
    std::unique_ptr<Proposal>
    TruncatedGaussianProposal<InternalMatrixType, InternalVectorType>::copyProposal() const {
        return std::make_unique<TruncatedGaussianProposal>(*this);
    }

    template<typename InternalMatrixType, typename InternalVectorType>
    VectorType
    TruncatedGaussianProposal<InternalMatrixType, InternalVectorType>::getState() const {
        return state;
    }

    template<typename InternalMatrixType, typename InternalVectorType>
    VectorType
    TruncatedGaussianProposal<InternalMatrixType, InternalVectorType>::getProposal() const {
        return proposal;
    }

    template<typename InternalMatrixType, typename InternalVectorType>
    std::vector<std::string>
    TruncatedGaussianProposal<InternalMatrixType, InternalVectorType>::getParameterNames() const {
        if (this->getStepSize().has_value()) {
            return {"step_size"};
        } else {
            return {};
        }
    }

    template<typename InternalMatrixType, typename InternalVectorType>
    std::any
    TruncatedGaussianProposal<InternalMatrixType, InternalVectorType>::getParameter(
            const ProposalParameter &parameter) const {
        throw std::invalid_argument("Can't get parameter which doesn't exist in " + this->getProposalName());
    }

    template<typename InternalMatrixType, typename InternalVectorType>
    std::string
    TruncatedGaussianProposal<InternalMatrixType, InternalVectorType>::getParameterType(
            const ProposalParameter &parameter) const {
        if (parameter == ProposalParameter::STEP_SIZE) {
            return "double";
        } else {
            throw std::invalid_argument("Can't get parameter which doesn't exist in " + this->getProposalName());
        }
    }

    template<typename InternalMatrixType, typename InternalVectorType>
    void TruncatedGaussianProposal<InternalMatrixType, InternalVectorType>::setParameter(
            const ProposalParameter &parameter, const std::any &value) {
        throw std::invalid_argument("Can't get parameter which doesn't exist in " + this->getProposalName());
    }

    template<typename InternalMatrixType, typename InternalVectorType>
    const MatrixType &
    TruncatedGaussianProposal<InternalMatrixType, InternalVectorType>::getA() const {
        return A;
    }

    template<typename InternalMatrixType, typename InternalVectorType>
    const VectorType &
    TruncatedGaussianProposal<InternalMatrixType, InternalVectorType>::getB() const {
        return b;
    }

    template<typename InternalMatrixType, typename InternalVectorType>
    ProposalStatistics &
    TruncatedGaussianProposal<InternalMatrixType, InternalVectorType>::getProposalStatistics() {
        return proposalStatistics;
    }

    template<typename InternalMatrixType, typename InternalVectorType>
    void
    TruncatedGaussianProposal<InternalMatrixType, InternalVectorType>::activateTrackingOfProposalStatistics() {
        isProposalInfosTrackingActive = true;
    }

    template<typename InternalMatrixType, typename InternalVectorType>
    void
    TruncatedGaussianProposal<InternalMatrixType, InternalVectorType>::disableTrackingOfProposalStatistics() {
        isProposalInfosTrackingActive = false;
    }

    template<typename InternalMatrixType, typename InternalVectorType>
    bool
    TruncatedGaussianProposal<InternalMatrixType, InternalVectorType>::isTrackingOfProposalStatisticsActivated() {
        return isProposalInfosTrackingActive;
    }

    template<typename InternalMatrixType, typename InternalVectorType>
    ProposalStatistics
    TruncatedGaussianProposal<InternalMatrixType, InternalVectorType>::getAndResetProposalStatistics() {
        ProposalStatistics newStatistic;
        std::swap(newStatistic, proposalStatistics);
        return newStatistic;
    }

    template<typename InternalMatrixType, typename InternalVectorType>
    std::unique_ptr<Model> TruncatedGaussianProposal<InternalMatrixType, InternalVectorType>::getModel() const {
        return Gaussian::copyModel();
    }

    template<typename InternalMatrixType, typename InternalVectorType>
    std::vector<std::string>
    TruncatedGaussianProposal<InternalMatrixType, InternalVectorType>::getDimensionNames() const {
        return Gaussian::getDimensionNames();
    }
}

#endif //HOPS_TRUNCATEDGAUSSIANPROPOSAL_HPP
