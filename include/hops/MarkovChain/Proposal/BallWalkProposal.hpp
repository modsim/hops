#ifndef HOPS_BALLWALKPROPOSAL_HPP
#define HOPS_BALLWALKPROPOSAL_HPP

#include "ChordStepDistributions.hpp"
#include <hops/MarkovChain/IsSetStepSizeAvailable.hpp>
#include <hops/RandomNumberGenerator/RandomNumberGenerator.hpp>
#include <random>
#include <hops/FileWriter/CsvWriter.hpp>

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
         * @param delta         The radius of the ball from which the proposal move is drawn
         */
        BallWalkProposal(MatrixType A, VectorType b, StateType currentState, typename MatrixType::Scalar delta = 1);

        void propose(RandomNumberGenerator &randomNumberGenerator);

        void acceptProposal();

        StateType getState() const;

        StateType getProposal() const;

        void setState(StateType newState);

        void setStepSize(typename MatrixType::Scalar stepSize);

        typename MatrixType::Scalar getStepSize() const;

        [[nodiscard]] typename MatrixType::Scalar calculateLogAcceptanceProbability() {
            return 1;
        }

        std::string getName();

    private:
        MatrixType A;
        VectorType b;
        StateType state;
        VectorType slacks;
        //VectorType inverseDistances;

        //long coordinateToUpdate = 0;
        //typename MatrixType::Scalar step = 0;
        typename MatrixType::Scalar forwardDistance;
        typename MatrixType::Scalar backwardDistance;

        VectorType step;
        VectorType radii;
        VectorType directions;
        typename MatrixType::Scalar delta;

        std::uniform_real_distribution<typename MatrixType::Scalar> uniform;
        std::normal_distribution<typename MatrixType::Scalar> normal;

        void drawFromBall(RandomNumberGenerator& randomNumberGenerator); 
    };

    template<typename MatrixType, typename VectorType>
    BallWalkProposal<MatrixType, VectorType>::BallWalkProposal(
            MatrixType A_,
            VectorType b_,
            VectorType currentState_, 
            typename MatrixType::Scalar delta_
        ) :
            A(std::move(A_)),
            b(std::move(b_)),
            state(std::move(currentState_)),
            delta(delta_)
    {
        step = VectorType::Zero(state.rows());
        radii = VectorType::Zero(state.rows()); 
        directions = VectorType::Zero(state.rows()); 
    }

    template<typename MatrixType, typename VectorType>
    void BallWalkProposal<MatrixType, VectorType>::drawFromBall(
            RandomNumberGenerator& randomNumberGenerator) {
        for (long i = 0; i < directions.rows(); ++i) {
            directions(i) = normal(randomNumberGenerator);
            radii(i) = std::pow(uniform(randomNumberGenerator), 1./directions.rows());
        }
        directions.normalize();

        for (long i = 0; i < step.rows(); ++i) {
            step(i) = delta * directions(i) * radii(i);
        }
    }

    template<typename MatrixType, typename VectorType>
    void BallWalkProposal<MatrixType, VectorType>::propose(
            RandomNumberGenerator& randomNumberGenerator) {
        // get step as a uniform random point in delta*B, where B is the unit ball
        drawFromBall(randomNumberGenerator);

        // compute if step would lead outside of the polytope. if so, step is the zero vector
        slacks = b - A*(state + step);
        
        if ((slacks.array() <= 0).any()) {
            for (long i = 0; i < step.rows(); ++i) {
                step(i) = 0;
            }
        }
    }

    template<typename MatrixType, typename VectorType>
    void
    BallWalkProposal<MatrixType, VectorType>::acceptProposal() {
        state += step;
    }

    template<typename MatrixType, typename VectorType>
    typename BallWalkProposal<MatrixType, VectorType>::StateType
    BallWalkProposal<MatrixType, VectorType>::getState() const {
        return state;
    }

    template<typename MatrixType, typename VectorType>
    typename BallWalkProposal<MatrixType, VectorType>::StateType
    BallWalkProposal<MatrixType, VectorType>::getProposal() const {
        StateType proposal = state + step;
        return proposal;
    }

    template<typename MatrixType, typename VectorType>
    void BallWalkProposal<MatrixType, VectorType>::setState(VectorType newState) {
        BallWalkProposal::state = std::move(newState);
    }

    template<typename MatrixType, typename VectorType>
    void BallWalkProposal<MatrixType, VectorType>::setStepSize(
            typename MatrixType::Scalar stepSize) {
        delta = stepSize;
    }

    template<typename MatrixType, typename VectorType>
    typename MatrixType::Scalar
    BallWalkProposal<MatrixType, VectorType>::getStepSize() const {
        return delta;
    }

    template<typename MatrixType, typename VectorType>
    std::string BallWalkProposal<MatrixType, VectorType>::getName() {
        return "BallWalk";
    }
}

#endif //HOPS_BALLWALKPROPOSAL_HPP
