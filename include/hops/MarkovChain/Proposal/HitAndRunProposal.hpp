#ifndef HOPS_HITANDRUNPROPOSAL_HPP
#define HOPS_HITANDRUNPROPOSAL_HPP

#include "ChordStepDistributions.hpp"
#include "../IsSetStepSizeAvailable.hpp"
#include "../../RandomNumberGenerator/RandomNumberGenerator.hpp"
#include <random>

namespace hops {
    template<typename MatrixType, typename VectorType, typename ChordStepDistribution = UniformStepDistribution<typename MatrixType::Scalar>, bool Precise = true>
    class HitAndRunProposal {
    public:
        using StateType = VectorType;

        HitAndRunProposal(MatrixType A, VectorType b, VectorType currentState);

        void propose(RandomNumberGenerator &randomNumberGenerator);

        void acceptProposal();

        StateType getState() const;

        StateType getProposal() const;

        void setState(StateType newState);

        void setStepSize(typename MatrixType::Scalar stepSize);

        [[nodiscard]] typename MatrixType::Scalar computeLogAcceptanceProbability() {
            return chordStepDistribution.computeInverseNormalizationConstant(0, backwardDistance, forwardDistance)
                   - chordStepDistribution.computeInverseNormalizationConstant(0, backwardDistance - step,
                                                                               forwardDistance - step);
        }

        [[nodiscard]] typename MatrixType::Scalar getStepSize() const;

        std::string getName();

    private:
        MatrixType A;
        VectorType b;
        StateType state;
        StateType proposal;
        VectorType slacks;
        VectorType inverseDistances;

        VectorType updateDirection;
        typename MatrixType::Scalar step = 0;
        ChordStepDistribution chordStepDistribution;
        std::normal_distribution<typename MatrixType::Scalar> normalDistribution;
        typename MatrixType::Scalar forwardDistance;
        typename MatrixType::Scalar backwardDistance;
    };

    template<typename MatrixType, typename VectorType, typename ChordStepDistribution, bool Precise>
    HitAndRunProposal<MatrixType, VectorType, ChordStepDistribution, Precise>::HitAndRunProposal(MatrixType A_,
                                                                                                 VectorType b_,
                                                                                                 VectorType currentState_)
            :
            A(std::move(A_)),
            b(std::move(b_)),
            state(std::move(currentState_)) {
        slacks = this->b - this->A * this->state;
        updateDirection = state;
    }

    template<typename MatrixType, typename VectorType, typename ChordStepDistribution, bool Precise>
    void HitAndRunProposal<MatrixType, VectorType, ChordStepDistribution, Precise>::propose(
            RandomNumberGenerator &randomNumberGenerator) {
        for (long i = 0; i < updateDirection.rows(); ++i) {
            updateDirection(i) = normalDistribution(randomNumberGenerator);
        }
        updateDirection.normalize();

        inverseDistances = (A * updateDirection).cwiseQuotient(slacks);
        forwardDistance = 1. / inverseDistances.maxCoeff();
        backwardDistance = 1. / inverseDistances.minCoeff();
        assert(backwardDistance < 0 && forwardDistance > 0);
        assert(((b - A * state).array() >= 0).all());

        step = chordStepDistribution.draw(randomNumberGenerator, backwardDistance, forwardDistance);
        proposal = state + updateDirection * step;
    }

    template<typename MatrixType, typename VectorType, typename ChordStepDistribution, bool Precise>
    void HitAndRunProposal<MatrixType, VectorType, ChordStepDistribution, Precise>::acceptProposal() {
        state = proposal;
        if (Precise) {
            slacks = b - A * state;
            if ((slacks.array() <= 0).any()) {
                throw std::runtime_error("Hit-and-Run sampled point outside of polytope.");
            }
        } else {
            slacks.noalias() -= A * updateDirection * step;
        }
    }

    template<typename MatrixType, typename VectorType, typename ChordStepDistribution, bool Precise>
    typename HitAndRunProposal<MatrixType, VectorType, ChordStepDistribution, Precise>::StateType
    HitAndRunProposal<MatrixType, VectorType, ChordStepDistribution, Precise>::getState() const {
        return state;
    }

    template<typename MatrixType, typename VectorType, typename ChordStepDistribution, bool Precise>
    typename HitAndRunProposal<MatrixType, VectorType, ChordStepDistribution, Precise>::StateType
    HitAndRunProposal<MatrixType, VectorType, ChordStepDistribution, Precise>::getProposal() const {
        return proposal;
    }

    template<typename MatrixType, typename VectorType, typename ChordStepDistribution, bool Precise>
    void HitAndRunProposal<MatrixType, VectorType, ChordStepDistribution, Precise>::setState(StateType newState) {
        HitAndRunProposal::state = std::move(newState);
        slacks = b - A * HitAndRunProposal::state;
    }

    template<typename MatrixType, typename VectorType, typename ChordStepDistribution, bool Precise>
    void HitAndRunProposal<MatrixType, VectorType, ChordStepDistribution, Precise>::setStepSize(
            typename MatrixType::Scalar stepSize) {
        if constexpr (IsSetStepSizeAvailable<ChordStepDistribution>::value) {
            chordStepDistribution.setStepSize(stepSize);
        } else {
            (void) stepSize;
            throw std::runtime_error("Step size not available.");
        }
    }

    template<typename MatrixType, typename VectorType, typename ChordStepDistribution, bool Precise>
    typename MatrixType::Scalar
    HitAndRunProposal<MatrixType, VectorType, ChordStepDistribution, Precise>::getStepSize() const {
        if constexpr (IsSetStepSizeAvailable<ChordStepDistribution>::value) {
            return chordStepDistribution.getStepSize();
        }
        throw std::runtime_error("Step size not available.");
    }

    template<typename MatrixType, typename VectorType, typename ChordStepDistribution, bool Precise>
    std::string HitAndRunProposal<MatrixType, VectorType, ChordStepDistribution, Precise>::getName() {
        return "Hit-and-Run";
    }
}

#endif //HOPS_HITANDRUNPROPOSAL_HPP
