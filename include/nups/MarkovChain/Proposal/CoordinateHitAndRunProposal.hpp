#ifndef NUPS_COORDINATEHITANDRUNPROPOSAL_HPP
#define NUPS_COORDINATEHITANDRUNPROPOSAL_HPP

#include <nups/RandomNumberGenerator/RandomNumberGenerator.hpp>
#include <random>

namespace nups {
    template<typename MatrixType, typename VectorType>
    class CoordinateHitAndRunProposal {
    public:
        using StateType = VectorType;

        /**
         * @brief Constructs Coordinate Hit and Run proposal mechanism on polytope defined as Ax<b.
         * @param A
         * @param b
         * @param currentState
         */
        CoordinateHitAndRunProposal(MatrixType A, VectorType b, StateType currentState);

        void propose(RandomNumberGenerator &randomNumberGenerator);

        void acceptProposal();

        StateType getState() const;

        StateType getProposal() const;

        void setState(StateType state);

    private:
        MatrixType A;
        VectorType b;
        StateType state;
        VectorType slacks;
        VectorType inverseDistances;

        long coordinateToUpdate = 0;
        double stepSize = 0;
        std::uniform_real_distribution<typename MatrixType::Scalar> stepSizeDistribution{0., 1.};
    };

    template<typename MatrixType, typename VectorType>
    CoordinateHitAndRunProposal<MatrixType, VectorType>::CoordinateHitAndRunProposal(MatrixType A_,
                                                                             VectorType b_,
                                                                             StateType currentState_)
            :
            A(std::move(A_)), b(std::move(b_)), state(std::move(currentState_)) {
        slacks = this->b - this->A * this->state;
    }

    template<typename MatrixType, typename VectorType>
    void CoordinateHitAndRunProposal<MatrixType, VectorType>::propose(RandomNumberGenerator &generator) {
        ++coordinateToUpdate %= state.rows();

        inverseDistances = A.col(coordinateToUpdate).cwiseQuotient(slacks);
        typename MatrixType::Scalar forwardDistance = 1. / inverseDistances.maxCoeff();
        typename MatrixType::Scalar backwardDistance = 1. / inverseDistances.minCoeff();
        assert(backwardDistance < 0 && forwardDistance > 0);
        assert(((b- A * state).array() > 0).all());

        stepSize = backwardDistance + (forwardDistance - backwardDistance) * stepSizeDistribution(generator);
    }

    template<typename MatrixType, typename VectorType>
    void
    CoordinateHitAndRunProposal<MatrixType, VectorType>::acceptProposal() {
        state(coordinateToUpdate) += stepSize;
        slacks.noalias() -= A.col(coordinateToUpdate) * stepSize;
        stepSize = 0;
    }

    template<typename MatrixType, typename VectorType>
    typename CoordinateHitAndRunProposal<MatrixType, VectorType>::StateType
    CoordinateHitAndRunProposal<MatrixType, VectorType>::getState() const {
        return state;
    }

    template<typename MatrixType, typename VectorType>
    typename CoordinateHitAndRunProposal<MatrixType, VectorType>::StateType
    CoordinateHitAndRunProposal<MatrixType, VectorType>::getProposal() const {
        StateType proposal = state;
        proposal(coordinateToUpdate) += stepSize;
        return proposal;
    }

    template<typename MatrixType, typename VectorType>
    void CoordinateHitAndRunProposal<MatrixType, VectorType>::setState(VectorType state) {
        CoordinateHitAndRunProposal::state = std::move(state);
        slacks = b - A * CoordinateHitAndRunProposal::state;
    }
}

#endif //NUPS_COORDINATEHITANDRUNPROPOSAL_HPP
