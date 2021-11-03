#ifndef HOPS_BALLWALK_HPP
#define HOPS_BALLWALK_HPP

#include <random>

#include <hops/Utility/VectorType.hpp>
#include <hops/RandomNumberGenerator/RandomNumberGenerator.hpp>

#include "IsSetStepSizeAvailable.hpp"
#include "Proposal.hpp"

namespace hops {
    template<typename InternalMatrixType, typename InternalVectorType>
    class BallWalk : public Proposal {
    public:
        /**
         * @brief Constructs Ballwalk proposal mechanism on polytope defined as Ax<b.
         * @param A
         * @param b
         * @param currentState
         * @param stepSize The radius of the ball from which the proposal move is drawn
         */
        BallWalk(InternalMatrixType A,
                 InternalVectorType b,
                 VectorType currentState,
                 double stepSize = 1);

        std::pair<double, VectorType> propose(RandomNumberGenerator &randomNumberGenerator) override;

        VectorType acceptProposal() override;

        void setState(VectorType newState) override;

        void setStepSize(double stepSize) override;

        [[nodiscard]] std::string getProposalName() const override;

        [[nodiscard]] std::optional<double> getStepSize() const override;


    private:
        [[nodiscard]] typename InternalMatrixType::Scalar computeLogAcceptanceProbability();

        InternalMatrixType A;
        InternalVectorType b;
        VectorType state;
        VectorType proposal;

        double stepSize;

        std::uniform_real_distribution<typename InternalMatrixType::Scalar> uniform;
        std::normal_distribution<typename InternalMatrixType::Scalar> normal;
    };

    template<typename InternalMatrixType, typename InternalVectorType>
    BallWalk<InternalMatrixType, InternalVectorType>::BallWalk(InternalMatrixType A_,
                                                               InternalVectorType b_,
                                                               VectorType currentState_,
                                                               double stepSize_) :
            A(std::move(A_)),
            b(std::move(b_)),
            state(std::move(currentState_)),
            proposal(this->state),
            stepSize(stepSize_) {}

    template<typename InternalMatrixType, typename InternalVectorType>
    std::pair<double, VectorType> BallWalk<InternalMatrixType, InternalVectorType>::propose(
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
    VectorType BallWalk<InternalMatrixType, InternalVectorType>::acceptProposal() {
        state.swap(proposal);
        return state;
    }

    template<typename InternalMatrixType, typename InternalVectorType>
    void BallWalk<InternalMatrixType, InternalVectorType>::setState(VectorType newState) {
        BallWalk::state = std::move(newState);
    }

    template<typename InternalMatrixType, typename InternalVectorType>
    void BallWalk<InternalMatrixType, InternalVectorType>::setStepSize(double stepSize) {
        BallWalk::stepSize = stepSize;
    }

    template<typename InternalMatrixType, typename InternalVectorType>
    std::optional<double> BallWalk<InternalMatrixType, InternalVectorType>::getStepSize() const {
        return stepSize;
    }

    template<typename InternalMatrixType, typename InternalVectorType>
    std::string BallWalk<InternalMatrixType, InternalVectorType>::getProposalName() const {
        return "BallWalk";
    }

    template<typename InternalMatrixType, typename InternalVectorType>
    typename InternalMatrixType::Scalar
    BallWalk<InternalMatrixType, InternalVectorType>::computeLogAcceptanceProbability() {
        bool isProposalInteriorPoint = ((A * proposal - b).array() < 0).all();
        if (!isProposalInteriorPoint) {
            return -std::numeric_limits<typename InternalMatrixType::Scalar>::infinity();
        }
        return 0;
    }
}


#endif //HOPS_BALLWALK_HPP
