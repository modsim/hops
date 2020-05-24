#ifndef HOPS_COORDINATEHITANDRUNPROPOSAL_HPP
#define HOPS_COORDINATEHITANDRUNPROPOSAL_HPP

#include "ChordStepDistributions.hpp"
#include <hops/MarkovChain/IsSetStepSizeAvailable.hpp>
#include <hops/RandomNumberGenerator/RandomNumberGenerator.hpp>
#include <random>
#include <hops/FileWriter/CsvWriter.hpp>

// TODO overrelaxed

namespace hops {
    template<typename MatrixType, typename VectorType, typename ChordStepDistribution = UniformStepDistribution<typename MatrixType::Scalar>>
    class CoordinateHitAndRunProposal {
    public:
        using StateType = VectorType;

        /**
         * @brief Constructs Coordinate Hit and Run proposal mechanism on polytope defined as Ax<b.
         * @param A
         * @param b
         * @param currentState
         */
        CoordinateHitAndRunProposal(MatrixType A, VectorType b, VectorType currentState);

        void propose(RandomNumberGenerator &randomNumberGenerator);

        void acceptProposal();

        StateType getState() const;

        StateType getProposal() const;

        void setState(StateType newState);

        void setStepSize(typename MatrixType::Scalar stepSize);

        typename MatrixType::Scalar getStepSize() const;

        [[nodiscard]] typename MatrixType::Scalar calculateLogAcceptanceProbability() {
            return chordStepDistribution.calculateInverseNormalizationConstant(0, backwardDistance, forwardDistance)
                   - chordStepDistribution.calculateInverseNormalizationConstant(0, backwardDistance - step,
                                                                                 forwardDistance - step);
        }

        std::string getName();

    private:
        MatrixType A;
        VectorType b;
        StateType state;
        VectorType slacks;
        VectorType inverseDistances;

        long coordinateToUpdate = 0;
        typename MatrixType::Scalar step = 0;
        ChordStepDistribution chordStepDistribution;
        typename MatrixType::Scalar forwardDistance;
        typename MatrixType::Scalar backwardDistance;
    };

    template<typename MatrixType, typename VectorType, typename ChordStepDistribution>
    CoordinateHitAndRunProposal<MatrixType, VectorType, ChordStepDistribution>::CoordinateHitAndRunProposal(
            MatrixType A_,
            VectorType b_,
            VectorType currentState_) :
            A(std::move(A_)),
            b(std::move(b_)),
            state(std::move(currentState_)) {
        slacks = this->b - this->A * this->state;
    }

    template<typename MatrixType, typename VectorType, typename ChordStepDistribution>
    void CoordinateHitAndRunProposal<MatrixType, VectorType, ChordStepDistribution>::propose(
            RandomNumberGenerator &randomNumberGenerator) {
        ++coordinateToUpdate %= state.rows();

        inverseDistances = A.col(coordinateToUpdate).cwiseQuotient(slacks);
        forwardDistance = 1. / inverseDistances.maxCoeff();
        backwardDistance = 1. / inverseDistances.minCoeff();
        assert(backwardDistance < 0 && forwardDistance > 0);
        assert(((b - A * state).array() > 0).all());

        step = chordStepDistribution.draw(randomNumberGenerator, backwardDistance, forwardDistance);
    }

    template<typename MatrixType, typename VectorType, typename ChordStepDistribution>
    void
    CoordinateHitAndRunProposal<MatrixType, VectorType, ChordStepDistribution>::acceptProposal() {
        state(coordinateToUpdate) += step;
        slacks.noalias() -= A.col(coordinateToUpdate) * step;
        step = 0;
    }

    template<typename MatrixType, typename VectorType, typename ChordStepDistribution>
    typename CoordinateHitAndRunProposal<MatrixType, VectorType, ChordStepDistribution>::StateType
    CoordinateHitAndRunProposal<MatrixType, VectorType, ChordStepDistribution>::getState() const {
        return state;
    }

    template<typename MatrixType, typename VectorType, typename ChordStepDistribution>
    typename CoordinateHitAndRunProposal<MatrixType, VectorType, ChordStepDistribution>::StateType
    CoordinateHitAndRunProposal<MatrixType, VectorType, ChordStepDistribution>::getProposal() const {
        StateType proposal = state;
        proposal(coordinateToUpdate) += step;
        return proposal;
    }

    template<typename MatrixType, typename VectorType, typename ChordStepDistribution>
    void CoordinateHitAndRunProposal<MatrixType, VectorType, ChordStepDistribution>::setState(VectorType newState) {
        CoordinateHitAndRunProposal::state = std::move(newState);
        slacks = b - A * CoordinateHitAndRunProposal::state;
    }

    template<typename MatrixType, typename VectorType, typename ChordStepDistribution>
    void CoordinateHitAndRunProposal<MatrixType, VectorType, ChordStepDistribution>::setStepSize(
            typename MatrixType::Scalar stepSize) {
        if constexpr (IsSetStepSizeAvailable<ChordStepDistribution>::value) {
            chordStepDistribution.setStepSize(stepSize);
        } else {
            throw std::runtime_error("Step size not available.");
        }
    }

    template<typename MatrixType, typename VectorType, typename ChordStepDistribution>
    typename MatrixType::Scalar
    CoordinateHitAndRunProposal<MatrixType, VectorType, ChordStepDistribution>::getStepSize() const {
        if constexpr (IsSetStepSizeAvailable<ChordStepDistribution>::value) {
            return chordStepDistribution.getStepSize();
        }
        throw std::runtime_error("Step size not available.");
    }

    template<typename MatrixType, typename VectorType, typename ChordStepDistribution>
    std::string CoordinateHitAndRunProposal<MatrixType, VectorType, ChordStepDistribution>::getName() {
        return "Coordinate Hit-and-Run";
    }
}

#endif //HOPS_COORDINATEHITANDRUNPROPOSAL_HPP
