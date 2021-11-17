#ifndef HOPS_COORDINATEHITANDRUNPROPOSAL_HPP
#define HOPS_COORDINATEHITANDRUNPROPOSAL_HPP

#include <random>

#include <hops/RandomNumberGenerator/RandomNumberGenerator.hpp>

#include "ChordStepDistributions.hpp"
#include "IsSetStepSizeAvailable.hpp"
#include "Proposal.hpp"

namespace hops {
    template<typename InternalMatrixType, typename InternalVectorType, typename ChordStepDistribution = UniformStepDistribution<typename InternalMatrixType::Scalar>>
    class CoordinateHitAndRunProposal : public Proposal {
    public:
        /**
         * @brief Constructs Coordinate Hit and Run proposal mechanism on polytope defined as Ax<b.
         * @param A
         * @param b
         * @param currentState
         */
        CoordinateHitAndRunProposal(InternalMatrixType A, InternalVectorType b, InternalVectorType currentState);

        std::pair<double, InternalVectorType> propose(RandomNumberGenerator &rng) override;

        VectorType acceptProposal() override;

        void setState(VectorType state) override;

        VectorType getState() const override;

        [[nodiscard]] std::optional<double> getStepSize() const override;

        void setStepSize(double stepSize) override;

        [[nodiscard]] bool hasStepSize() const override;

        [[nodiscard]] std::string getProposalName() const override;

        [[nodiscard]] std::unique_ptr<Proposal> deepCopy() const override;

    private:
        [[nodiscard]] typename InternalMatrixType::Scalar computeLogAcceptanceProbability();

        InternalMatrixType A;
        InternalVectorType b;
        VectorType state;
        VectorType proposal;
        InternalVectorType slacks;
        InternalVectorType inverseDistances;

        long coordinateToUpdate = 0;
        typename InternalMatrixType::Scalar step = 0;
        ChordStepDistribution chordStepDistribution;
        typename InternalMatrixType::Scalar forwardDistance;
        typename InternalMatrixType::Scalar backwardDistance;
    };

    template<typename InternalMatrixType, typename InternalVectorType, typename ChordStepDistribution>
    CoordinateHitAndRunProposal<InternalMatrixType, InternalVectorType, ChordStepDistribution>::CoordinateHitAndRunProposal(
            InternalMatrixType A_,
            InternalVectorType b_,
            InternalVectorType currentState_) :
            A(std::move(A_)),
            b(std::move(b_)),
            state(std::move(currentState_)),
            proposal(this->state) {
        slacks = this->b - this->A * this->state;
    }

    template<typename InternalMatrixType, typename InternalVectorType, typename ChordStepDistribution>
    std::pair<double, InternalVectorType>
    CoordinateHitAndRunProposal<InternalMatrixType, InternalVectorType, ChordStepDistribution>::propose(
            RandomNumberGenerator &rng) {
        proposal(coordinateToUpdate) = state(coordinateToUpdate);
        ++coordinateToUpdate %= state.rows();

        inverseDistances = A.col(coordinateToUpdate).cwiseQuotient(slacks);
        forwardDistance = 1. / inverseDistances.maxCoeff();
        backwardDistance = 1. / inverseDistances.minCoeff();
        assert(backwardDistance < 0 && forwardDistance > 0);
        assert(((b - A * state).array() > 0).all());

        step = chordStepDistribution.draw(rng, backwardDistance, forwardDistance);

        proposal(coordinateToUpdate) += step;

        return {computeLogAcceptanceProbability(), proposal};
    }

    template<typename InternalMatrixType, typename InternalVectorType, typename ChordStepDistribution>
    VectorType
    CoordinateHitAndRunProposal<InternalMatrixType, InternalVectorType, ChordStepDistribution>::acceptProposal() {
        state(coordinateToUpdate) += step;
        slacks.noalias() -= A.col(coordinateToUpdate) * step;
        return state;
    }

    template<typename InternalMatrixType, typename InternalVectorType, typename ChordStepDistribution>
    void CoordinateHitAndRunProposal<InternalMatrixType, InternalVectorType, ChordStepDistribution>::setState(
            VectorType newState) {
        CoordinateHitAndRunProposal::state = std::move(newState);
        CoordinateHitAndRunProposal::proposal = CoordinateHitAndRunProposal::state;
        slacks = b - A * CoordinateHitAndRunProposal::state;
    }

    template<typename InternalMatrixType, typename InternalVectorType, typename ChordStepDistribution>
    std::optional<double>
    CoordinateHitAndRunProposal<InternalMatrixType, InternalVectorType, ChordStepDistribution>::getStepSize() const {
        if constexpr (IsSetStepSizeAvailable<ChordStepDistribution>::value) {
            return chordStepDistribution.getStepSize();
        }
        return std::nullopt;
    }

    template<typename InternalMatrixType, typename InternalVectorType, typename ChordStepDistribution>
    void CoordinateHitAndRunProposal<InternalMatrixType, InternalVectorType, ChordStepDistribution>::setStepSize(
            double stepSize) {
        if constexpr (IsSetStepSizeAvailable<ChordStepDistribution>::value) {
            chordStepDistribution.setStepSize(stepSize);
        }
    }

    template<typename InternalMatrixType, typename InternalVectorType, typename ChordStepDistribution>
    std::string
    CoordinateHitAndRunProposal<InternalMatrixType, InternalVectorType, ChordStepDistribution>::getProposalName() const {
        return "CoordinateHitAndRun";
    }

    template<typename InternalMatrixType, typename InternalVectorType, typename ChordStepDistribution>
    typename InternalMatrixType::Scalar
    CoordinateHitAndRunProposal<InternalMatrixType, InternalVectorType, ChordStepDistribution>::computeLogAcceptanceProbability() {
        return chordStepDistribution.computeInverseNormalizationConstant(0, backwardDistance, forwardDistance)
               - chordStepDistribution.computeInverseNormalizationConstant(0, backwardDistance - step,
                                                                           forwardDistance - step);
    }

    template<typename InternalMatrixType, typename InternalVectorType, typename ChordStepDistribution>
    bool CoordinateHitAndRunProposal<InternalMatrixType, InternalVectorType, ChordStepDistribution>::hasStepSize() const {
        if constexpr (IsSetStepSizeAvailable<ChordStepDistribution>::value) {
            return true;
        }
        return false;
    }

    template<typename InternalMatrixType, typename InternalVectorType, typename ChordStepDistribution>
    std::unique_ptr<Proposal>
    CoordinateHitAndRunProposal<InternalMatrixType, InternalVectorType, ChordStepDistribution>::deepCopy() const {
        return std::make_unique<CoordinateHitAndRunProposal>(*this);
    }

    template<typename InternalMatrixType, typename InternalVectorType, typename ChordStepDistribution>
    VectorType
    CoordinateHitAndRunProposal<InternalMatrixType, InternalVectorType, ChordStepDistribution>::getState() const {
        return state;
    }
}

#endif //HOPS_COORDINATEHITANDRUNPROPOSAL_HPP
