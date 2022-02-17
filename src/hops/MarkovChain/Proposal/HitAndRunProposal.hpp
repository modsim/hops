#ifndef HOPS_HITANDRUNPROPOSAL_HPP
#define HOPS_HITANDRUNPROPOSAL_HPP

#include <random>

#include <hops/RandomNumberGenerator/RandomNumberGenerator.hpp>
#include <hops/Utility/MatrixType.hpp>
#include <hops/Utility/StringUtility.hpp>
#include <hops/Utility/VectorType.hpp>

#include "ChordStepDistributions.hpp"
#include "IsSetStepSizeAvailable.hpp"
#include "Proposal.hpp"

namespace hops {
    template<typename InternalMatrixType, typename InternalVectorType, typename ChordStepDistribution = UniformStepDistribution<double>, bool Precise = false>
    class HitAndRunProposal : public Proposal {
    public:
        HitAndRunProposal(InternalMatrixType A, InternalVectorType b, InternalVectorType currentState,
                          double stepSize = 1);

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
        VectorType state;
        VectorType proposal;
        ProposalStatistics proposalStatistics;

        InternalMatrixType A;
        InternalVectorType b;
        InternalVectorType slacks;
        InternalVectorType inverseDistances;

        InternalVectorType updateDirection;
        double step = 0;
        ChordStepDistribution chordStepDistribution;
        std::normal_distribution<double> normalDistribution;
        double forwardDistance = 0;
        double backwardDistance = 0;

        bool isProposalInfosTrackingActive = false;
    };

    template<typename InternalMatrixType, typename InternalVectorType, typename ChordStepDistribution, bool Precise>
    double
    HitAndRunProposal<InternalMatrixType, InternalVectorType, ChordStepDistribution, Precise>::computeLogAcceptanceProbability() {
        double detailedBalanceState = chordStepDistribution.computeInverseNormalizationConstant(0, backwardDistance,
                                                                                                forwardDistance);
        double detailedBalanceProposal = chordStepDistribution.computeInverseNormalizationConstant(0, backwardDistance -
                                                                                                      step,
                                                                                                   forwardDistance -
                                                                                                   step);
        return detailedBalanceState - detailedBalanceProposal;
    }

    template<typename InternalMatrixType, typename InternalVectorType, typename ChordStepDistribution, bool Precise>
    HitAndRunProposal<InternalMatrixType, InternalVectorType, ChordStepDistribution, Precise>::HitAndRunProposal(
            InternalMatrixType A_,
            InternalVectorType b_,
            InternalVectorType currentState_,
            double stepSize) :
            A(std::move(A_)),
            b(std::move(b_)),
            state(std::move(currentState_)) {
        if (((b - A * state).array() < 0).any()) {
            throw std::invalid_argument("Starting point outside polytope always gives constant Markov chain.");
        }
        slacks = this->b - this->A * this->state;
        updateDirection = state;
        setStepSize(stepSize);
    }

    template<typename InternalMatrixType, typename InternalVectorType, typename ChordStepDistribution, bool Precise>
    VectorType &
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

        if (isProposalInfosTrackingActive) {
            proposalStatistics.appendInfo("backwardDistance", forwardDistance);
            proposalStatistics.appendInfo("forwardDistance", backwardDistance);
        }

        step = chordStepDistribution.draw(rng, backwardDistance, forwardDistance);
        proposal = state + updateDirection * step;

        return proposal;
    }

    template<typename InternalMatrixType, typename InternalVectorType, typename ChordStepDistribution, bool Precise>
    VectorType &
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

        return proposal;
    }

    template<typename InternalMatrixType, typename InternalVectorType, typename ChordStepDistribution, bool Precise>
    VectorType &
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
        proposalStatistics = ProposalStatistics();
        return state;
    }

    template<typename InternalMatrixType, typename InternalVectorType, typename ChordStepDistribution, bool Precise>
    void HitAndRunProposal<InternalMatrixType, InternalVectorType, ChordStepDistribution, Precise>::setState(
            const VectorType &newState) {
        assert(((b - A * newState).array() >= 0).all());
        if (((b - A * newState).array() < 0).any()) {
            throw std::invalid_argument("Starting point outside polytope always gives constant Markov chain.");
        }
        HitAndRunProposal::state = newState;
        HitAndRunProposal::proposal = HitAndRunProposal::state;
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
    HitAndRunProposal<InternalMatrixType, InternalVectorType, ChordStepDistribution, Precise>::copyProposal() const {
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
    std::vector<std::string>
    HitAndRunProposal<InternalMatrixType, InternalVectorType, ChordStepDistribution, Precise>::getParameterNames() const {
        if (this->getStepSize().has_value()) {
            return {"step_size"};
        }
        return {};
    }

    template<typename InternalMatrixType, typename InternalVectorType, typename ChordStepDistribution, bool Precise>
    std::any HitAndRunProposal<InternalMatrixType, InternalVectorType, ChordStepDistribution, Precise>::getParameter(
            const ProposalParameter &parameter) const {
        if (parameter == ProposalParameter::STEP_SIZE) {
            std::optional<double> s = this->getStepSize();
            if (s) {
                return std::any(s.value());
            }
        }
        throw std::invalid_argument("Can't get parameter which doesn't exist in " + this->getProposalName());
    }

    template<typename InternalMatrixType, typename InternalVectorType, typename ChordStepDistribution, bool Precise>
    std::string
    HitAndRunProposal<InternalMatrixType, InternalVectorType, ChordStepDistribution, Precise>::getParameterType(
            const ProposalParameter &parameter) const {
        if (parameter == ProposalParameter::STEP_SIZE) {
            return "double";
        }
        throw std::invalid_argument("Can't get parameter which doesn't exist in " + this->getProposalName());
    }

    template<typename InternalMatrixType, typename InternalVectorType, typename ChordStepDistribution, bool Precise>
    void HitAndRunProposal<InternalMatrixType, InternalVectorType, ChordStepDistribution, Precise>::setParameter(
            const ProposalParameter &parameter, const std::any &value) {
        if (parameter == ProposalParameter::STEP_SIZE) {
            setStepSize(std::any_cast<double>(value));
        } else {
            throw std::invalid_argument("Can't get parameter which doesn't exist in " + this->getProposalName());
        }
    }

    template<typename InternalMatrixType, typename InternalVectorType, typename ChordStepDistribution, bool Precise>
    const MatrixType& 
    HitAndRunProposal<InternalMatrixType, InternalVectorType, ChordStepDistribution, Precise>::getA() const {
        return A;
    }

    template<typename InternalMatrixType, typename InternalVectorType, typename ChordStepDistribution, bool Precise>
    const VectorType& 
    HitAndRunProposal<InternalMatrixType, InternalVectorType, ChordStepDistribution, Precise>::getB() const {
        return b;
    }

    template<typename InternalMatrixType, typename InternalVectorType, typename ChordStepDistribution, bool Precise>
    ProposalStatistics &
    HitAndRunProposal<InternalMatrixType, InternalVectorType, ChordStepDistribution, Precise>::getProposalStatistics() {
        return proposalStatistics;
    }

    template<typename InternalMatrixType, typename InternalVectorType, typename ChordStepDistribution, bool Precise>
    void
    HitAndRunProposal<InternalMatrixType, InternalVectorType, ChordStepDistribution, Precise>::activateTrackingOfProposalStatistics() {
        isProposalInfosTrackingActive = true;
    }

    template<typename InternalMatrixType, typename InternalVectorType, typename ChordStepDistribution, bool Precise>
    void
    HitAndRunProposal<InternalMatrixType, InternalVectorType, ChordStepDistribution, Precise>::disableTrackingOfProposalStatistics() {
        isProposalInfosTrackingActive = false;
    }

    template<typename InternalMatrixType, typename InternalVectorType, typename ChordStepDistribution, bool Precise>
    bool
    HitAndRunProposal<InternalMatrixType, InternalVectorType, ChordStepDistribution, Precise>::isTrackingOfProposalStatisticsActivated() {
        return isProposalInfosTrackingActive;
    }

    template<typename InternalMatrixType, typename InternalVectorType, typename ChordStepDistribution, bool Precise>
    ProposalStatistics
    HitAndRunProposal<InternalMatrixType, InternalVectorType, ChordStepDistribution, Precise>::getAndResetProposalStatistics() {
        ProposalStatistics newStatistic;
        std::swap(newStatistic, proposalStatistics);
        return newStatistic;
    }
}

#endif //HOPS_HITANDRUNPROPOSAL_HPP
