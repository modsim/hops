#ifndef NUPS_COORDINATEHITANDRUNROUNDEDPROPOSAL_HPP
#define NUPS_COORDINATEHITANDRUNROUNDEDPROPOSAL_HPP

#include <nups/RandomNumberGenerator/RandomNumberGenerator.hpp>
#include <random>

namespace nups {
    template<typename MatrixType, typename VectorType>
    class CoordinateHitAndRunRoundedProposal {
    public:
        using StateType = VectorType;

        /**
         * @brief Constructs Coordinate Hit and Run on polytope defined as Ax<b, where the proposals are rounded
         *        by the maximum volume ellipsoid.
         * @param A
         * @param roundingTransformation Cholesky L of Maximum Volume Ellipsoid.
         * @param b
         * @param currentState
         */
        CoordinateHitAndRunRoundedProposal(MatrixType A, MatrixType roundingTransformation, VectorType b,
                                           StateType currentState);

        void propose(RandomNumberGenerator &randomNumberGenerator);

        void acceptProposal();

        StateType getState() const;

        StateType getProposal() const;

        void setState(StateType state);

    private:
        MatrixType A;
        MatrixType ATimesRounding;
        MatrixType normalizedRounding;
        VectorType b;
        StateType state;
        VectorType slacks;
        VectorType inverseDistances;

        long coordinateToUpdate = 0;
        double stepSize = 0;
        std::uniform_real_distribution<typename MatrixType::Scalar> stepSizeDistribution{0., 1.};
    };

    template<typename MatrixType, typename VectorType>
    CoordinateHitAndRunRoundedProposal<MatrixType, VectorType>::CoordinateHitAndRunRoundedProposal(MatrixType A_,
                                                                                                   MatrixType roundingTransform,
                                                                                                   VectorType b_,
                                                                                                   StateType currentState_)
            :
            A(std::move(A_)), b(std::move(b_)), state(std::move(currentState_)) {
        assert(((b - A * state).array() > 0).all());
        slacks = b - A * state;
        normalizedRounding.resize(roundingTransform.rows(), roundingTransform.cols());
        for (long i = 0; i < roundingTransform.cols(); ++i) {
            normalizedRounding.col(i) = roundingTransform.col(i).normalized();
        }

        ATimesRounding = CoordinateHitAndRunRoundedProposal::A * normalizedRounding;
    }

    template<typename MatrixType, typename VectorType>
    void CoordinateHitAndRunRoundedProposal<MatrixType, VectorType>::acceptProposal() {
        state += normalizedRounding.col(coordinateToUpdate) * stepSize;
        slacks.noalias() -= ATimesRounding.col(coordinateToUpdate) * stepSize;
    }

    template<typename MatrixType, typename VectorType>
    void CoordinateHitAndRunRoundedProposal<MatrixType, VectorType>::propose(RandomNumberGenerator &generator) {
        ++coordinateToUpdate %= state.rows();

        inverseDistances = ATimesRounding.col(coordinateToUpdate).cwiseQuotient(slacks);
        typename MatrixType::Scalar forwardDistance = 1. / inverseDistances.maxCoeff();
        typename MatrixType::Scalar backwardDistance = 1. / inverseDistances.minCoeff();
        assert(backwardDistance < 0 && forwardDistance > 0);
        assert(((b - A * state).array() > 0).all());

        stepSize = backwardDistance + (forwardDistance - backwardDistance) * stepSizeDistribution(generator);
    }

    template<typename MatrixType, typename VectorType>
    typename CoordinateHitAndRunRoundedProposal<MatrixType, VectorType>::StateType
    CoordinateHitAndRunRoundedProposal<MatrixType, VectorType>::getState() const {
        return state;
    }

    template<typename MatrixType, typename VectorType>
    typename CoordinateHitAndRunRoundedProposal<MatrixType, VectorType>::StateType
    CoordinateHitAndRunRoundedProposal<MatrixType, VectorType>::getProposal() const {
        StateType proposal = state;
        proposal(coordinateToUpdate) += stepSize;
        return proposal;
    }

    template<typename MatrixType, typename VectorType>
    void CoordinateHitAndRunRoundedProposal<MatrixType, VectorType>::setState(VectorType state) {
        CoordinateHitAndRunRoundedProposal::state = std::move(state);
        slacks = b - A * CoordinateHitAndRunRoundedProposal::state;
    }
}

#endif //NUPS_COORDINATEHITANDRUNROUNDEDPROPOSAL_HPP
