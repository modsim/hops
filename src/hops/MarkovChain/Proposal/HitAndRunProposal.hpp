#ifndef HOPS_HITANDRUNPROPOSAL_HPP
#define HOPS_HITANDRUNPROPOSAL_HPP

#include <random>

#include <hops/RandomNumberGenerator/RandomNumberGenerator.hpp>

#include "ChordStepDistributions.hpp"
#include "IsSetStepSizeAvailable.hpp"
#include "Proposal.hpp"

namespace hops {
    template<typename InternalMatrixType, typename InternalVectorType, typename ChordStepDistribution = UniformStepDistribution<typename InternalMatrixType::Scalar>, bool Precise = false>
    class HitAndRunProposal : public Proposal {
    public:
        HitAndRunProposal(InternalMatrixType A, InternalVectorType b, InternalVectorType currentState);

        std::pair<double, VectorType> propose(RandomNumberGenerator &rng) override;

        std::pair<double, VectorType> propose(RandomNumberGenerator &rng, const std::vector<int> &activeSubspace);

        VectorType acceptProposal() override;

        void setState(VectorType state) override;

        [[nodiscard]] VectorType getState() const override;

        [[nodiscard]] VectorType getProposal() const override;

        void setParameter(ProposalParameterName parameterName, const std::any &value) override;

        [[nodiscard]] std::optional<double> getStepSize() const;

        void setStepSize(double stepSize);

        [[nodiscard]] bool hasStepSize() const override;

        [[nodiscard]] std::string getProposalName() const override;

        [[nodiscard]] std::unique_ptr<Proposal> deepCopy() const override;

        [[nodiscard]] double computeLogAcceptanceProbability();

    private:
        VectorType state;
        VectorType proposal;

        InternalMatrixType A;
        InternalVectorType b;
        InternalVectorType slacks;
        InternalVectorType inverseDistances;

        InternalVectorType updateDirection;
        typename InternalMatrixType::Scalar step = 0;
        ChordStepDistribution chordStepDistribution;
        std::normal_distribution<typename InternalMatrixType::Scalar> normalDistribution;
        typename InternalMatrixType::Scalar forwardDistance;
        typename InternalMatrixType::Scalar backwardDistance;
    };

    template<typename InternalMatrixType, typename InternalVectorType, typename ChordStepDistribution, bool Precise>
    double
    HitAndRunProposal<InternalMatrixType, InternalVectorType, ChordStepDistribution, Precise>::computeLogAcceptanceProbability() {
        return chordStepDistribution.computeInverseNormalizationConstant(0, backwardDistance, forwardDistance)
               - chordStepDistribution.computeInverseNormalizationConstant(0, backwardDistance - step,
                                                                           forwardDistance - step);

    }

    template<typename InternalMatrixType, typename InternalVectorType, typename ChordStepDistribution, bool Precise>
    HitAndRunProposal<InternalMatrixType, InternalVectorType, ChordStepDistribution, Precise>::HitAndRunProposal(
            InternalMatrixType A_,
            InternalVectorType b_,
            InternalVectorType currentState_)
            :
            A(std::move(A_)),
            b(std::move(b_)),
            state(std::move(currentState_)) {
        slacks = this->b - this->A * this->state;
        updateDirection = state;
    }

    template<typename InternalMatrixType, typename InternalVectorType, typename ChordStepDistribution, bool Precise>
    std::pair<double, VectorType>
    HitAndRunProposal<InternalMatrixType, InternalVectorType, ChordStepDistribution, Precise>::propose(
            RandomNumberGenerator &rng) {
        for (long i = 0; i < updateDirection.rows(); ++i) {
            updateDirection(i) = normalDistribution(rng);
        }
        updateDirection.normalize();

        inverseDistances = (A * updateDirection).cwiseQuotient(slacks);
        forwardDistance = 1. / inverseDistances.maxCoeff();
        backwardDistance = 1. / inverseDistances.minCoeff();
        assert(backwardDistance <= 0 && forwardDistance >= 0);
        assert(((b - A * state).array() >= 0).all());

        step = chordStepDistribution.draw(rng, backwardDistance, forwardDistance);
        proposal = state + updateDirection * step;

        return {computeLogAcceptanceProbability(), proposal};
    }

    template<typename InternalMatrixType, typename InternalVectorType, typename ChordStepDistribution, bool Precise>
    std::pair<double, VectorType>
    HitAndRunProposal<InternalMatrixType, InternalVectorType, ChordStepDistribution, Precise>::propose(
            RandomNumberGenerator &rng, const std::vector<int> &activeSubspace) {
        for (long i = 0; i < updateDirection.rows(); ++i) {
            if (activeSubspace[i]) {
                updateDirection(i) = normalDistribution(rng);
            } else {
                updateDirection[i] = 0;
            }
        }
        updateDirection.normalize();

        inverseDistances = (A * updateDirection).cwiseQuotient(slacks);
        // Inverse distance are potentially nan due to default values on the boundary of the polytope.
        // Replaces nan because nan should not influence the distances.
        inverseDistances = inverseDistances
                .array()
                .unaryExpr([](double value) { return std::isfinite(value) ? value : 0.; })
                .matrix();

        forwardDistance = 1. / inverseDistances.maxCoeff();
        backwardDistance = 1. / inverseDistances.minCoeff();
        assert(backwardDistance <= 0 && forwardDistance >= 0);
        assert(((b - A * state).array() >= 0).all());

        step = chordStepDistribution.draw(rng, backwardDistance, forwardDistance);
        proposal = state + updateDirection * step;
        assert(((b - A * proposal).array() >= 0).all());

        return {computeLogAcceptanceProbability(), proposal};
    }

    template<typename InternalMatrixType, typename InternalVectorType, typename ChordStepDistribution, bool Precise>
    VectorType
    HitAndRunProposal<InternalMatrixType, InternalVectorType, ChordStepDistribution, Precise>::acceptProposal() {
        state = proposal;
        if (Precise) {
            slacks = b - A * state;
            if ((slacks.array() < 0).any()) {
                throw std::runtime_error("Hit-and-Run sampled point outside of polytope.");
            }
        } else {
            slacks.noalias() -= A * updateDirection * step;
        }
        return state;
    }

    template<typename InternalMatrixType, typename InternalVectorType, typename ChordStepDistribution, bool Precise>
    void HitAndRunProposal<InternalMatrixType, InternalVectorType, ChordStepDistribution, Precise>::setState(
            VectorType newState) {
        assert(((b - A * newState).array() >= 0).all());
        HitAndRunProposal::state = std::move(newState);
        HitAndRunProposal::proposal = state;
        slacks = b - A * HitAndRunProposal::state;
    }

    template<typename InternalMatrixType, typename InternalVectorType, typename ChordStepDistribution, bool Precise>
    std::optional<double>
    HitAndRunProposal<InternalMatrixType, InternalVectorType, ChordStepDistribution, Precise>::getStepSize() const {
        if constexpr (IsSetStepSizeAvailable<ChordStepDistribution>::value) {
            return chordStepDistribution.getStepSize();
        }
        return std::nullopt;
    }

    template<typename InternalMatrixType, typename InternalVectorType, typename ChordStepDistribution, bool Precise>
    void HitAndRunProposal<InternalMatrixType, InternalVectorType, ChordStepDistribution, Precise>::setStepSize(
            double stepSize) {
        if constexpr (IsSetStepSizeAvailable<ChordStepDistribution>::value) {
            chordStepDistribution.setStepSize(stepSize);
        }
    }

    template<typename InternalMatrixType, typename InternalVectorType, typename ChordStepDistribution, bool Precise>
    bool
    HitAndRunProposal<InternalMatrixType, InternalVectorType, ChordStepDistribution, Precise>::hasStepSize() const {
        if constexpr (IsSetStepSizeAvailable<ChordStepDistribution>::value) {
            return true;
        }
        return false;
    }

    template<typename InternalMatrixType, typename InternalVectorType, typename ChordStepDistribution, bool Precise>
    std::string
    HitAndRunProposal<InternalMatrixType, InternalVectorType, ChordStepDistribution, Precise>::getProposalName() const {
        return "HitAndRun";
    }

    template<typename InternalMatrixType, typename InternalVectorType, typename ChordStepDistribution, bool Precise>
    std::unique_ptr<Proposal>
    HitAndRunProposal<InternalMatrixType, InternalVectorType, ChordStepDistribution, Precise>::deepCopy() const {
        return std::make_unique<HitAndRunProposal>(*this);
    }

    template<typename InternalMatrixType, typename InternalVectorType, typename ChordStepDistribution, bool Precise>
    VectorType
    HitAndRunProposal<InternalMatrixType, InternalVectorType, ChordStepDistribution, Precise>::getState() const {
        return state;
    }

    template<typename InternalMatrixType, typename InternalVectorType, typename ChordStepDistribution, bool Precise>
    VectorType
    HitAndRunProposal<InternalMatrixType, InternalVectorType, ChordStepDistribution, Precise>::getProposal() const {
        return proposal;
    }

    template<typename InternalMatrixType, typename InternalVectorType, typename ChordStepDistribution, bool Precise>
    void HitAndRunProposal<InternalMatrixType, InternalVectorType, ChordStepDistribution, Precise>::setParameter(
            ProposalParameterName parameterName, const std::any &value) {
        switch (parameterName) {
            case ProposalParameterName::STEP_SIZE: {
                setStepSize(std::any_cast<double>(value));
                break;
            }
            default:
                throw std::invalid_argument("Can't set parameter which doesn't exist in HitAndRunProposal.");
        }
    }
}

#endif //HOPS_HITANDRUNPROPOSAL_HPP
