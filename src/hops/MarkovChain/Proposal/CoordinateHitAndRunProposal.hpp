#ifndef HOPS_COORDINATEHITANDRUNPROPOSAL_HPP
#define HOPS_COORDINATEHITANDRUNPROPOSAL_HPP

#include <random>

#include <hops/RandomNumberGenerator/RandomNumberGenerator.hpp>
#include <hops/Utility/MatrixType.hpp>
#include <hops/Utility/StringUtility.hpp>
#include <hops/Utility/VectorType.hpp>

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
        CoordinateHitAndRunProposal(InternalMatrixType A, InternalVectorType b, InternalVectorType currentState, double stepSize = 1);

        VectorType &propose(RandomNumberGenerator &rng) override;

        VectorType &propose(RandomNumberGenerator &rng, const std::vector<int> &activeSubspace) override;

        VectorType &acceptProposal() override;

        void setState(const VectorType &state) override;

        [[nodiscard]] VectorType getState() const override;

        [[nodiscard]] VectorType getProposal() const override;

        [[nodiscard]] std::vector<std::string> getParameterNames() const override;

        [[nodiscard]] std::any getParameter(const ProposalParameter &parameter) const override;

        [[nodiscard]] std::string getParameterType(const ProposalParameter &parameter) const override;

        void setParameter(const ProposalParameter &parameter, const std::any &value) override;

        [[nodiscard]] std::optional<double> getStepSize() const;

        void setStepSize(double stepSize);

        [[nodiscard]] bool hasStepSize() const override;

        [[nodiscard]] std::string getProposalName() const override;

        [[nodiscard]] std::unique_ptr<Proposal> copyProposal() const override;

        [[nodiscard]] double computeLogAcceptanceProbability() override;

        [[nodiscard]] const MatrixType& getA() const override;

        [[nodiscard]] const VectorType& getB() const override;

        ProposalStatistics & getProposalStatistics() override;

        void activateTrackingOfProposalStatistics() override;

        void disableTrackingOfProposalStatistics() override;

        bool isTrackingOfProposalStatisticsActivated() override;

        ProposalStatistics getAndResetProposalStatistics() override;

    private:
        InternalMatrixType A;
        InternalVectorType b;
        VectorType state;
        VectorType proposal;
        InternalVectorType slacks;
        InternalVectorType inverseDistances;
        ProposalStatistics proposalStatistics;

        long coordinateToUpdate = 0;
        typename InternalMatrixType::Scalar step = 0;
        ChordStepDistribution chordStepDistribution;
        typename InternalMatrixType::Scalar forwardDistance = 0;
        typename InternalMatrixType::Scalar backwardDistance = 0;

        bool isProposalInfosTrackingActive = false;
    };

    template<typename InternalMatrixType, typename InternalVectorType, typename ChordStepDistribution>
    CoordinateHitAndRunProposal<InternalMatrixType, InternalVectorType, ChordStepDistribution>::CoordinateHitAndRunProposal(InternalMatrixType A_,
                                                                                                                            InternalVectorType b_,
                                                                                                                            InternalVectorType currentState_,
                                                                                                                            double stepSize) :
            A(std::move(A_)),
            b(std::move(b_)),
            state(std::move(currentState_)),
            proposal(this->state) {
        slacks = this->b - this->A * this->state;
        setStepSize(stepSize);
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
        if (isProposalInfosTrackingActive) {
            proposalStatistics.appendInfo("backwardDistance", forwardDistance);
            proposalStatistics.appendInfo("forwardDistance", backwardDistance);
        }

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
        double detailedBalanceState = chordStepDistribution.computeInverseNormalizationConstant(0, backwardDistance,
                                                                                                forwardDistance);
        double detailedBalanceProposal = chordStepDistribution.computeInverseNormalizationConstant(0, backwardDistance -
                                                                                                      step,
                                                                                                   forwardDistance -
                                                                                                   step);
        return detailedBalanceState - detailedBalanceProposal;
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
    CoordinateHitAndRunProposal<InternalMatrixType, InternalVectorType, ChordStepDistribution>::copyProposal() const {
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
            const ProposalParameter &parameter) const {
        if (parameter == ProposalParameter::STEP_SIZE) {
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
            const ProposalParameter &parameter) const {
        if (parameter == ProposalParameter::STEP_SIZE) {
            return "double";
        } else {
            throw std::invalid_argument("Can't get parameter which doesn't exist in " + this->getProposalName());
        }
    }

    template<typename InternalMatrixType, typename InternalVectorType, typename ChordStepDistribution>
    void CoordinateHitAndRunProposal<InternalMatrixType, InternalVectorType, ChordStepDistribution>::setParameter(
            const ProposalParameter &parameter, const std::any &value) {
        if (parameter == ProposalParameter::STEP_SIZE) {
            setStepSize(std::any_cast<double>(value));
        } else {
            throw std::invalid_argument("Can't get parameter which doesn't exist in " + this->getProposalName());
        }
    }

    template<typename InternalMatrixType, typename InternalVectorType, typename ChordStepDistribution>
    const MatrixType& 
    CoordinateHitAndRunProposal<InternalMatrixType, InternalVectorType, ChordStepDistribution>::getA() const {
        return A;
      }

    template<typename InternalMatrixType, typename InternalVectorType, typename ChordStepDistribution>
    const VectorType& 
    CoordinateHitAndRunProposal<InternalMatrixType, InternalVectorType, ChordStepDistribution>::getB() const {
        return b;
    }

    template<typename InternalMatrixType, typename InternalVectorType, typename ChordStepDistribution>
    ProposalStatistics &
    CoordinateHitAndRunProposal<InternalMatrixType, InternalVectorType, ChordStepDistribution>::getProposalStatistics() {
        return proposalStatistics;
    }

    template<typename InternalMatrixType, typename InternalVectorType, typename ChordStepDistribution>
    void
    CoordinateHitAndRunProposal<InternalMatrixType, InternalVectorType, ChordStepDistribution>::activateTrackingOfProposalStatistics() {
        isProposalInfosTrackingActive = true;
    }

    template<typename InternalMatrixType, typename InternalVectorType, typename ChordStepDistribution>
    void
    CoordinateHitAndRunProposal<InternalMatrixType, InternalVectorType, ChordStepDistribution>::disableTrackingOfProposalStatistics() {
        isProposalInfosTrackingActive = false;
    }

    template<typename InternalMatrixType, typename InternalVectorType, typename ChordStepDistribution>
    bool
    CoordinateHitAndRunProposal<InternalMatrixType, InternalVectorType, ChordStepDistribution>::isTrackingOfProposalStatisticsActivated() {
        return isProposalInfosTrackingActive;
    }

    template<typename InternalMatrixType, typename InternalVectorType, typename ChordStepDistribution>
    ProposalStatistics
    CoordinateHitAndRunProposal<InternalMatrixType, InternalVectorType, ChordStepDistribution>::getAndResetProposalStatistics() {
        ProposalStatistics newStatistic;
        std::swap(newStatistic, proposalStatistics);
        return newStatistic;
    }
}

#endif //HOPS_COORDINATEHITANDRUNPROPOSAL_HPP
