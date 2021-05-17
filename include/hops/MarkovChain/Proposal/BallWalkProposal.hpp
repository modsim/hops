#ifndef HOPS_BALLWALKPROPOSAL_HPP
#define HOPS_BALLWALKPROPOSAL_HPP

#include "../IsSetStepSizeAvailable.hpp"
#include "../../RandomNumberGenerator/RandomNumberGenerator.hpp"
#include "../../FileWriter/CsvWriter.hpp"
#include <random>

namespace hops {
    template<typename MatrixType, typename VectorType>
    class BallWalkProposal {
    public:
        using StateType = VectorType;

        /**
         * @brief Constructs Ballwalk proposal mechanism on polytope defined as Ax<b.
         * @param A
         * @param b
         * @param currentState
         * @param stepSize         The radius of the ball from which the proposal move is drawn
         */
        BallWalkProposal(MatrixType A, VectorType b, StateType currentState, typename MatrixType::Scalar stepSize = 1);

        void propose(RandomNumberGenerator &randomNumberGenerator);

        void acceptProposal();

        StateType getState() const;

        StateType getProposal() const;

        void setState(StateType newState);

        void setStepSize(typename MatrixType::Scalar stepSize);

        typename MatrixType::Scalar getStepSize() const;

        [[nodiscard]] typename MatrixType::Scalar calculateLogAcceptanceProbability() {
            bool isProposalInteriorPoint = ((A * proposal - b).array() < 0).all();
            if (!isProposalInteriorPoint) {
                return -std::numeric_limits<typename MatrixType::Scalar>::infinity();
            }
            return 0;
        }

        std::string getName();

    private:
        MatrixType A;
        VectorType b;
        StateType state;
        StateType proposal;

        typename MatrixType::Scalar stepSize;

        std::uniform_real_distribution<typename MatrixType::Scalar> uniform;
        std::normal_distribution<typename MatrixType::Scalar> normal;
    };

    template<typename MatrixType, typename VectorType>
    BallWalkProposal<MatrixType, VectorType>::BallWalkProposal(MatrixType A_,
                                                               VectorType b_,
                                                               VectorType currentState_,
                                                               typename MatrixType::Scalar stepSize_) :
            A(std::move(A_)),
            b(std::move(b_)),
            state(std::move(currentState_)),
            proposal(this->state),
            stepSize(stepSize_) {
        // nothing to do
    }

    template<typename MatrixType, typename VectorType>
    void BallWalkProposal<MatrixType, VectorType>::propose(
            RandomNumberGenerator &randomNumberGenerator) {
        // Creates proposal on Ballsurface
        for (long i = 0; i < proposal.rows(); ++i) {
            proposal(i) = normal(randomNumberGenerator);
        }
        proposal.normalize();

        // Scale proposal to radius of Ball
        proposal.noalias() = stepSize*proposal;

        // Scales proposal into Ball
        proposal.noalias() = std::pow(uniform(randomNumberGenerator), 1. / proposal.rows()) * proposal;

        proposal.noalias() += state;
    }

    template<typename MatrixType, typename VectorType>
    void
    BallWalkProposal<MatrixType, VectorType>::acceptProposal() {
        state.swap(proposal);
    }

    template<typename MatrixType, typename VectorType>
    typename BallWalkProposal<MatrixType, VectorType>::StateType
    BallWalkProposal<MatrixType, VectorType>::getState() const {
        return state;
    }

    template<typename MatrixType, typename VectorType>
    typename BallWalkProposal<MatrixType, VectorType>::StateType
    BallWalkProposal<MatrixType, VectorType>::getProposal() const {
        return proposal;
    }

    template<typename MatrixType, typename VectorType>
    void BallWalkProposal<MatrixType, VectorType>::setState(VectorType newState) {
        BallWalkProposal::state = std::move(newState);
    }

    template<typename MatrixType, typename VectorType>
    void BallWalkProposal<MatrixType, VectorType>::setStepSize(
            typename MatrixType::Scalar newStepSize) {
        stepSize = newStepSize;
    }

    template<typename MatrixType, typename VectorType>
    typename MatrixType::Scalar
    BallWalkProposal<MatrixType, VectorType>::getStepSize() const {
        return stepSize;
    }

    template<typename MatrixType, typename VectorType>
    std::string BallWalkProposal<MatrixType, VectorType>::getName() {
        return "Ball Walk";
    }
}

#endif //HOPS_BALLWALKPROPOSAL_HPP
