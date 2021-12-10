#ifndef HOPS_COORDINATEHITANDRUNPROPOSAL_HPP
#define HOPS_COORDINATEHITANDRUNPROPOSAL_HPP

#include <random>

#include <hops/RandomNumberGenerator/RandomNumberGenerator.hpp>
#include <hops/Utility/StringUtility.hpp>

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

        VectorType &propose(RandomNumberGenerator &rng) override;

        VectorType &propose(RandomNumberGenerator &rng, const std::vector<int> &activeSubspace) override;

        VectorType &acceptProposal() override;

        void setState(const VectorType &state) override;

        [[nodiscard]] VectorType getState() const override;

        [[nodiscard]] VectorType getProposal() const override;

        [[nodiscard]] std::vector<std::string> getDimensionNames() const override;

        [[nodiscard]] std::vector<std::string> getParameterNames() const override;

        [[nodiscard]] std::any getParameter(const std::string &parameterName) const override;

        [[nodiscard]] std::string getParameterType(const std::string &name) const override;

        void setParameter(const std::string &parameterName, const std::any &value) override;

        [[nodiscard]] std::optional<double> getStepSize() const;

        void setStepSize(double stepSize);

        [[nodiscard]] bool hasStepSize() const override;

        [[nodiscard]] std::string getProposalName() const override;

        [[nodiscard]] std::unique_ptr<Proposal> deepCopy() const override;

        [[nodiscard]] double computeLogAcceptanceProbability() override;

    private:
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
    VectorType &CoordinateHitAndRunProposal<InternalMatrixType, InternalVectorType, ChordStepDistribution>::propose(
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

        return proposal;
    }

    template<typename InternalMatrixType, typename InternalVectorType, typename ChordStepDistribution>
    VectorType &CoordinateHitAndRunProposal<InternalMatrixType, InternalVectorType, ChordStepDistribution>::propose(
            RandomNumberGenerator &rng, const std::vector<int> &activeSubspace) {
        proposal(coordinateToUpdate) = state(coordinateToUpdate);
        // Check that at least some spaces are active
        assert(std::accumulate(activeSubspace.begin(), activeSubspace.end(), 0) > 0);
        do {
            ++coordinateToUpdate %= state.rows();
        } while (activeSubspace[coordinateToUpdate] == 0);

        inverseDistances = A.col(coordinateToUpdate).cwiseQuotient(slacks);
        // Inverse distance are potentially nan due to default values on the boundary of the polytope.
        // Replaces nan because nan should not influence the distances.
        inverseDistances = inverseDistances
                .array()
                .unaryExpr([](double value) { return std::isfinite(value) ? value : 0.; })
                .matrix();
        forwardDistance = 1. / inverseDistances.maxCoeff();
        backwardDistance = 1. / inverseDistances.minCoeff();
        assert(backwardDistance < 0 && forwardDistance > 0);
        assert(((b - A * state).array() >= 0).all());

        step = chordStepDistribution.draw(rng, backwardDistance, forwardDistance);

        proposal(coordinateToUpdate) += step;

        return proposal;
    }

    template<typename InternalMatrixType, typename InternalVectorType, typename ChordStepDistribution>
    VectorType&
    CoordinateHitAndRunProposal<InternalMatrixType, InternalVectorType, ChordStepDistribution>::acceptProposal() {
        state(coordinateToUpdate) += step;
        slacks.noalias() -= A.col(coordinateToUpdate) * step;
        return state;
    }

    template<typename InternalMatrixType, typename InternalVectorType, typename ChordStepDistribution>
    void CoordinateHitAndRunProposal<InternalMatrixType, InternalVectorType, ChordStepDistribution>::setState(
            const VectorType &newState) {
        CoordinateHitAndRunProposal::state = newState;
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
    double
    CoordinateHitAndRunProposal<InternalMatrixType, InternalVectorType, ChordStepDistribution>::computeLogAcceptanceProbability() {
        return chordStepDistribution.computeInverseNormalizationConstant(0, backwardDistance, forwardDistance)
               - chordStepDistribution.computeInverseNormalizationConstant(0, backwardDistance - step,
                                                                           forwardDistance - step);
    }

    template<typename InternalMatrixType, typename InternalVectorType, typename ChordStepDistribution>
    bool
    CoordinateHitAndRunProposal<InternalMatrixType, InternalVectorType, ChordStepDistribution>::hasStepSize() const {
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

    template<typename InternalMatrixType, typename InternalVectorType, typename ChordStepDistribution>
    VectorType
    CoordinateHitAndRunProposal<InternalMatrixType, InternalVectorType, ChordStepDistribution>::getProposal() const {
        return proposal;
    }

    template<typename InternalMatrixType, typename InternalVectorType, typename ChordStepDistribution>
    std::vector<std::string>
    CoordinateHitAndRunProposal<InternalMatrixType, InternalVectorType, ChordStepDistribution>::getParameterNames() const {
        if(this->getStepSize().has_value()) {
            return {"step_size"};
        }
        else {
            return {};
        }
    }

    template<typename InternalMatrixType, typename InternalVectorType, typename ChordStepDistribution>
    std::any CoordinateHitAndRunProposal<InternalMatrixType, InternalVectorType, ChordStepDistribution>::getParameter(
            const std::string &parameterName) const {
        std::string lowerCaseParameterName = toLowerCase(parameterName);
        if (lowerCaseParameterName == "step_size") {
            std::optional<double> s = this->getStepSize();
            if (s) {
                return std::any(s.value());
            }
        }
        throw std::invalid_argument("Can't get parameter which doesn't exist in " + this->getProposalName());
    }

    template<typename InternalMatrixType, typename InternalVectorType, typename ChordStepDistribution>
    std::string
    CoordinateHitAndRunProposal<InternalMatrixType, InternalVectorType, ChordStepDistribution>::getParameterType(
            const std::string &name) const {
        std::string lowerCaseParameterName = toLowerCase(name);
        if (lowerCaseParameterName == "step_size") {
            return "double";
        } else {
            throw std::invalid_argument("Can't get parameter which doesn't exist in " + this->getProposalName());
        }
    }

    template<typename InternalMatrixType, typename InternalVectorType, typename ChordStepDistribution>
    void CoordinateHitAndRunProposal<InternalMatrixType, InternalVectorType, ChordStepDistribution>::setParameter(
            const std::string &parameterName, const std::any &value) {
        std::string lowerCaseParameterName = toLowerCase(parameterName);
        if (lowerCaseParameterName == "step_size") {
            setStepSize(std::any_cast<double>(value));
        } else {
            throw std::invalid_argument("Can't get parameter which doesn't exist in " + this->getProposalName());
        }
    }

    template<typename InternalMatrixType, typename InternalVectorType, typename ChordStepDistribution>
    std::vector<std::string>
    CoordinateHitAndRunProposal<InternalMatrixType, InternalVectorType, ChordStepDistribution>::getDimensionNames() const {
        std::vector<std::string> names;
        for (long i = 0; i < state.rows(); ++i) {
            names.emplace_back("x_" + std::to_string(i));
        }
        return names;
    }
}

#endif //HOPS_COORDINATEHITANDRUNPROPOSAL_HPP
