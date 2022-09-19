#ifndef HOPS_HITANDRUNPROPOSAL_HPP
#define HOPS_HITANDRUNPROPOSAL_HPP

#include <optional>
#include <random>

#include "hops/RandomNumberGenerator/RandomNumberGenerator.hpp"
#include "hops/Utility/DefaultDimensionNames.hpp"
#include "hops/Utility/MatrixType.hpp"
#include "hops/Utility/StringUtility.hpp"
#include "hops/Utility/VectorType.hpp"

#include "ChordStepDistributions.hpp"
#include "IsGetStepSizeAvailable.hpp"
#include "IsSetStepSizeAvailable.hpp"
#include "Proposal.hpp"

namespace hops {
    template<typename InternalMatrixType, typename InternalVectorType, typename ChordStepDistribution = UniformStepDistribution<double>, bool Precise = false>
    class HitAndRunProposal : public Proposal {
    public:
        HitAndRunProposal(InternalMatrixType A, InternalVectorType b, InternalVectorType currentState,
                          double stepSize = 1);

        VectorType &propose(RandomNumberGenerator &rng) override;

        VectorType &propose(RandomNumberGenerator &rng, const Eigen::VectorXd &activeIndices) override;

        VectorType &acceptProposal() override;

        void setState(const VectorType &state) override;

        void setProposal(const VectorType &newProposal) override;

        [[nodiscard]] VectorType getState() const override;

        [[nodiscard]] VectorType getProposal() const override;

        void setDimensionNames(const std::vector<std::string> &names) override;

        [[nodiscard]] std::vector<std::string> getDimensionNames() const override;

        [[nodiscard]] std::vector<std::string> getParameterNames() const override;

        [[nodiscard]] std::any getParameter(const ProposalParameter &parameter) const override;

        [[nodiscard]] std::string getParameterType(const ProposalParameter &parameter) const override;

        void setParameter(const ProposalParameter &parameter, const std::any &value) override;

        [[nodiscard]] std::optional<double> getStepSize() const override;

        void setStepSize(double stepSize);

        [[nodiscard]] static bool hasStepSize();

        [[nodiscard]] std::string getProposalName() const override;

        [[nodiscard]] std::unique_ptr<Proposal> copyProposal() const override;

        [[nodiscard]] double computeLogAcceptanceProbability() override;

        [[nodiscard]] const MatrixType &getA() const override;

        [[nodiscard]] const VectorType &getB() const override;

        bool isSymmetric() const override;

    private:
        VectorType state;
        VectorType proposal;

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

        std::vector<std::string> dimensionNames;
    };

    template<typename InternalMatrixType, typename InternalVectorType, typename ChordStepDistribution, bool Precise>
    double
    HitAndRunProposal<InternalMatrixType, InternalVectorType, ChordStepDistribution, Precise>::computeLogAcceptanceProbability() {
        if constexpr (IsGetStepSizeAvailable<ChordStepDistribution>::value) {
            double stepSize = chordStepDistribution.getStepSize();
            double detailedBalanceState = chordStepDistribution.computeInverseNormalizationConstant(stepSize,
                                                                                                    backwardDistance,
                                                                                                    forwardDistance);
            double detailedBalanceProposal = chordStepDistribution.computeInverseNormalizationConstant(stepSize,
                                                                                                       backwardDistance -
                                                                                                       step,
                                                                                                       forwardDistance -
                                                                                                       step);
            return detailedBalanceState - detailedBalanceProposal;
        } else {
            double detailedBalanceState = chordStepDistribution.computeInverseNormalizationConstant(1,
                                                                                                    backwardDistance,
                                                                                                    forwardDistance);
            double detailedBalanceProposal = chordStepDistribution.computeInverseNormalizationConstant(1,
                                                                                                       backwardDistance -
                                                                                                       step,
                                                                                                       forwardDistance -
                                                                                                       step);
            return detailedBalanceState - detailedBalanceProposal;
        }
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
        proposal = state;
        setStepSize(stepSize);
        this->dimensionNames = createDefaultDimensionNames(this->state.rows());
    }

    template<typename InternalMatrixType, typename InternalVectorType, typename ChordStepDistribution, bool Precise>
    VectorType &
    HitAndRunProposal<InternalMatrixType, InternalVectorType, ChordStepDistribution, Precise>::propose(
            RandomNumberGenerator &rng) {
        for (long i = 0; i < updateDirection.rows(); ++i) {
            this->updateDirection(i) = normalDistribution(rng);
        }
        this->updateDirection.normalize();

        this->inverseDistances = (this->A * this->updateDirection).cwiseQuotient(this->slacks);
        this->inverseDistances = this->inverseDistances
                .array()
                .unaryExpr([](double value) { return std::isnan(value) ? 0. : value; })
                .matrix();
        this->forwardDistance = 1. / this->inverseDistances.maxCoeff();
        this->backwardDistance = 1. / this->inverseDistances.minCoeff();
        if (this->forwardDistance < 0) {
            // forward direction is unconstrained
            this->forwardDistance = std::numeric_limits<typename InternalMatrixType::Scalar>::infinity();
        }
        if (this->backwardDistance > 0) {
            // backward direction is unconstrained
            this->backwardDistance = -std::numeric_limits<typename InternalMatrixType::Scalar>::infinity();
        }
        assert(((this->b - this->A * this->state).array() >= 0).all());

        this->step = this->chordStepDistribution.draw(rng, backwardDistance, forwardDistance);
        this->proposal = this->state + this->updateDirection * this->step;

        return this->proposal;
    }

    template<typename InternalMatrixType, typename InternalVectorType, typename ChordStepDistribution, bool Precise>
    VectorType &
    HitAndRunProposal<InternalMatrixType, InternalVectorType, ChordStepDistribution, Precise>::propose(
            RandomNumberGenerator &rng, const Eigen::VectorXd &activeIndices) {
        slacks = this->b - this->A * this->state;
        updateDirection.setZero();
        assert(activeIndices.sum() > 0);
        for (long i = 0; i < activeIndices.rows(); ++i) {
            updateDirection(i) = (activeIndices(i) != 0) ? normalDistribution(rng) : 0;
        }
        updateDirection.normalize();

        inverseDistances = (A * updateDirection).cwiseQuotient(slacks);
        // Inverse distance are potentially nan due to default values on the boundary of the polytope.
        // Replaces nan because nan should not influence the distances.
        this->inverseDistances = this->inverseDistances
                .array()
                .unaryExpr([](double value) { return std::isnan(value) ? 0. : value; })
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
        proposal = state;
        if constexpr (Precise) {
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
            const VectorType &newState) {
        if (((b - A * newState).array() < 0).any()) {
            std::stringstream str;
            str << (b - A * newState).transpose() << std::endl;
            str << "state was\n" << newState.transpose() << std::endl;
            throw std::invalid_argument(
                    "Starting point outside polytope always gives constant Markov chain.\n" + str.str());
        }
        HitAndRunProposal::state = newState;
        HitAndRunProposal::proposal = HitAndRunProposal::state;
        slacks = b - A * HitAndRunProposal::state;
    }

    template<typename InternalMatrixType, typename InternalVectorType, typename ChordStepDistribution, bool Precise>
    void HitAndRunProposal<InternalMatrixType, InternalVectorType, ChordStepDistribution, Precise>::setProposal(
            const VectorType &newProposal) {
        HitAndRunProposal::proposal = newProposal;

        step = (proposal - state).coeff(0);
        updateDirection = (proposal - state).normalized();

        inverseDistances = (A * updateDirection).cwiseQuotient(slacks);
        forwardDistance = 1. / inverseDistances.maxCoeff();
        backwardDistance = 1. / inverseDistances.minCoeff();
        if (forwardDistance < 0) {
            // forward direction is unconstrained
            forwardDistance = std::numeric_limits<typename InternalMatrixType::Scalar>::infinity();
        }
        if (backwardDistance > 0) {
            // backward direction is unconstrained
            backwardDistance = -std::numeric_limits<typename InternalMatrixType::Scalar>::infinity();
        }
    }

    template<typename InternalMatrixType, typename InternalVectorType, typename ChordStepDistribution, bool Precise>
    std::optional<double>
    HitAndRunProposal<InternalMatrixType, InternalVectorType, ChordStepDistribution, Precise>::getStepSize() const {
        if constexpr (IsGetStepSizeAvailable<ChordStepDistribution>::value) {
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
    HitAndRunProposal<InternalMatrixType, InternalVectorType, ChordStepDistribution, Precise>::hasStepSize() {
        if constexpr (IsGetStepSizeAvailable<ChordStepDistribution>::value) {
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
            return {ProposalParameterName[static_cast<int>(ProposalParameter::STEP_SIZE)]};
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
    const MatrixType &
    HitAndRunProposal<InternalMatrixType, InternalVectorType, ChordStepDistribution, Precise>::getA() const {
        return A;
    }

    template<typename InternalMatrixType, typename InternalVectorType, typename ChordStepDistribution, bool Precise>
    const VectorType &
    HitAndRunProposal<InternalMatrixType, InternalVectorType, ChordStepDistribution, Precise>::getB() const {
        return b;
    }

    template<typename InternalMatrixType, typename InternalVectorType, typename ChordStepDistribution, bool Precise>
    bool
    HitAndRunProposal<InternalMatrixType, InternalVectorType, ChordStepDistribution, Precise>::isSymmetric() const {
        // As soon as there is a step size the polytope borders and normalization will make the proposal asymmetric.
        if (getStepSize()) {
            return false;
        }
        return true;
    }

    template<typename InternalMatrixType, typename InternalVectorType, typename ChordStepDistribution, bool Precise>
    void HitAndRunProposal<InternalMatrixType, InternalVectorType, ChordStepDistribution, Precise>::setDimensionNames(
            const std::vector<std::string> &names) {
        dimensionNames = names;
    }

    template<typename InternalMatrixType, typename InternalVectorType, typename ChordStepDistribution, bool Precise>
    std::vector<std::string>
    HitAndRunProposal<InternalMatrixType, InternalVectorType, ChordStepDistribution, Precise>::getDimensionNames() const {
        return dimensionNames;
    }
}

#endif //HOPS_HITANDRUNPROPOSAL_HPP
