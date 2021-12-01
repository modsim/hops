#ifndef HOPS_BALLWALKPROPOSAL_HPP
#define HOPS_BALLWALKPROPOSAL_HPP

#include <random>

#include <hops/Utility/VectorType.hpp>
#include <hops/RandomNumberGenerator/RandomNumberGenerator.hpp>

#include "IsSetStepSizeAvailable.hpp"
#include "Proposal.hpp"

namespace hops {
    template<typename InternalMatrixType, typename InternalVectorType>
    class BallWalkProposal : public Proposal {
    public:
        /**
         * @brief Constructs Ballwalk proposal mechanism on polytope defined as Ax<b.
         * @param A
         * @param b
         * @param currentState
         * @param stepSize The radius of the ball from which the proposal move is drawn
         */
        BallWalkProposal(InternalMatrixType A, InternalVectorType b, VectorType currentState, double stepSize = 1);

        std::pair<double, VectorType> propose(RandomNumberGenerator &randomNumberGenerator) override;

        VectorType acceptProposal() override;

        void setState(VectorType newState) override;

        [[nodiscard]] VectorType getState() const override;

        VectorType getProposal() const override;

        void setStepSize(double stepSize);

        [[nodiscard]] std::string getProposalName() const override;

        [[nodiscard]] std::optional<double> getStepSize() const;

        [[nodiscard]] bool hasStepSize() const override;

        [[nodiscard]] std::unique_ptr<Proposal> deepCopy() const override;

        [[nodiscard]] double computeLogAcceptanceProbability();

    private:
        InternalMatrixType A;
        InternalVectorType b;
        VectorType state;
        VectorType proposal;

        double stepSize;

        std::uniform_real_distribution<typename InternalMatrixType::Scalar> uniform;
        std::normal_distribution<typename InternalMatrixType::Scalar> normal;
    };

    template<typename InternalMatrixType, typename InternalVectorType>
    BallWalkProposal<InternalMatrixType, InternalVectorType>::BallWalkProposal(InternalMatrixType A_,
                                                                               InternalVectorType b_,
                                                                               VectorType currentState_,
                                                                               double stepSize_) :
            A(std::move(A_)),
            b(std::move(b_)),
            state(std::move(currentState_)),
            proposal(this->state),
            stepSize(stepSize_) {}

    template<typename InternalMatrixType, typename InternalVectorType>
    std::pair<double, VectorType> BallWalkProposal<InternalMatrixType, InternalVectorType>::propose(
            RandomNumberGenerator &randomNumberGenerator) {
        // Creates proposal on Ballsurface
        for (long i = 0; i < proposal.rows(); ++i) {
            proposal(i) = normal(randomNumberGenerator);
        }
        proposal.normalize();
        // Scales proposal to radius of Ball
        proposal.noalias() = stepSize * proposal;
        // Scales proposal into Ball
        proposal.noalias() = std::pow(uniform(randomNumberGenerator), 1. / proposal.rows()) * proposal;
        proposal.noalias() += state;

        return {computeLogAcceptanceProbability(), proposal};
    }

    template<typename InternalMatrixType, typename InternalVectorType>
    VectorType BallWalkProposal<InternalMatrixType, InternalVectorType>::acceptProposal() {
        state.swap(proposal);
        return state;
    }

    template<typename InternalMatrixType, typename InternalVectorType>
    void BallWalkProposal<InternalMatrixType, InternalVectorType>::setState(VectorType newState) {
        BallWalkProposal::state = std::move(newState);
    }

    template<typename InternalMatrixType, typename InternalVectorType>
    void BallWalkProposal<InternalMatrixType, InternalVectorType>::setStepSize(double stepSize) {
        BallWalkProposal::stepSize = stepSize;
    }

    template<typename InternalMatrixType, typename InternalVectorType>
    std::optional<double> BallWalkProposal<InternalMatrixType, InternalVectorType>::getStepSize() const {
        return stepSize;
    }

    template<typename InternalMatrixType, typename InternalVectorType>
    std::string BallWalkProposal<InternalMatrixType, InternalVectorType>::getProposalName() const {
        return "BallWalkProposal";
    }

    template<typename InternalMatrixType, typename InternalVectorType>
    double BallWalkProposal<InternalMatrixType, InternalVectorType>::computeLogAcceptanceProbability() {
        bool isProposalInteriorPoint = ((A * proposal - b).array() < 0).all();
        if (!isProposalInteriorPoint) {
            return -std::numeric_limits<typename InternalMatrixType::Scalar>::infinity();
        }
        return 0;
    }

    template<typename InternalMatrixType, typename InternalVectorType>
    bool BallWalkProposal<InternalMatrixType, InternalVectorType>::hasStepSize() const {
        return true;
    }

    template<typename InternalMatrixType, typename InternalVectorType>
    std::unique_ptr<Proposal> BallWalkProposal<InternalMatrixType, InternalVectorType>::deepCopy() const {
        return std::make_unique<BallWalkProposal>(*this);
    }

    template<typename InternalMatrixType, typename InternalVectorType>
    VectorType BallWalkProposal<InternalMatrixType, InternalVectorType>::getState() const {
        return state;
    }

    template<typename InternalMatrixType, typename InternalVectorType>
    VectorType BallWalkProposal<InternalMatrixType, InternalVectorType>::getProposal() const {
        return proposal;
    }
}


#endif //HOPS_BALLWALKPROPOSAL_HPP
