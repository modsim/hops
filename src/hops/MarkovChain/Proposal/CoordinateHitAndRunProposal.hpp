#ifndef HOPS_COORDINATEHITANDRUNPROPOSAL_HPP
#define HOPS_COORDINATEHITANDRUNPROPOSAL_HPP

#include <optional>
#include <random>

#include "hops/RandomNumberGenerator/RandomNumberGenerator.hpp"
#include "hops/Utility/DefaultDimensionNames.hpp"
#include "hops/Utility/MatrixType.hpp"
#include "hops/Utility/StringUtility.hpp"
#include "hops/Utility/VectorType.hpp"

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
        CoordinateHitAndRunProposal(InternalMatrixType A,
                                    InternalVectorType b,
                                    InternalVectorType currentState,
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
        InternalMatrixType A;
        InternalVectorType b;
        VectorType state;
        VectorType proposal;
        InternalVectorType slacks;
        InternalVectorType proposalSlacks;
        InternalVectorType inverseDistances;
        bool shouldRecomputeSlacks = false;
        double detailedBalance = 0;

        long coordinateToUpdate = 0;
        typename InternalMatrixType::Scalar step = 0;
        ChordStepDistribution chordStepDistribution;
        typename InternalMatrixType::Scalar forwardDistance = 0;
        typename InternalMatrixType::Scalar backwardDistance = 0;

        std::vector<std::string> dimensionNames;
    };

    template<typename InternalMatrixType, typename InternalVectorType, typename ChordStepDistribution>
    CoordinateHitAndRunProposal<InternalMatrixType, InternalVectorType, ChordStepDistribution>::CoordinateHitAndRunProposal(
            InternalMatrixType A_,
            InternalVectorType b_,
            InternalVectorType currentState_,
            double stepSize) :
            A(std::move(A_)),
            b(std::move(b_)),
            state(std::move(currentState_)),
            proposal(this->state) {
        if (((b - A * state).array() < 0).any()) {
            throw std::invalid_argument("Starting point outside polytope always gives constant Markov chain.");
        }
        slacks = this->b - this->A * this->state;
        setStepSize(stepSize);

        this->dimensionNames = createDefaultDimensionNames(this->state.rows());
    }

    template<typename InternalMatrixType, typename InternalVectorType, typename ChordStepDistribution>
    VectorType &CoordinateHitAndRunProposal<InternalMatrixType, InternalVectorType, ChordStepDistribution>::propose(
            RandomNumberGenerator &rng) {
        proposal(coordinateToUpdate) = state(coordinateToUpdate);
        ++coordinateToUpdate %= state.rows();

        inverseDistances = A.col(coordinateToUpdate).cwiseQuotient(slacks);
        // Inverse distance are potentially nan due to default values on the boundary of the polytope.
        // Replaces nan because nan should not influence the distances.
        this->inverseDistances = this->inverseDistances
                .array()
                .unaryExpr([](double value) { return std::isnan(value) ? 0. : value; })
                .matrix();
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

        assert(((b - A * state).array() >= 0).all());

        step = chordStepDistribution.draw(rng, backwardDistance, forwardDistance);

        proposal(coordinateToUpdate) += step;

        return proposal;
    }

    template<typename InternalMatrixType, typename InternalVectorType, typename ChordStepDistribution>
    VectorType &CoordinateHitAndRunProposal<InternalMatrixType, InternalVectorType, ChordStepDistribution>::propose(
            RandomNumberGenerator &rng, const Eigen::VectorXd &activeIndices) {
        this->shouldRecomputeSlacks = true;
        if (activeIndices.sum() <= 0) {
            // No active subspaces
            detailedBalance = 0;
            return state;
        }

        proposal = state;
        proposalSlacks = slacks;
        for (long i = 0; i < activeIndices.rows(); ++i) {
            if (activeIndices(i) == 0) { continue; }
            inverseDistances = A.col(i).cwiseQuotient(proposalSlacks);
            // Inverse distance are potentially nan due to default values on the boundary of the polytope.
            // Replaces nan because nan should not influence the distances.
            this->inverseDistances = this->inverseDistances
                    .array()
                    .unaryExpr([](double value) { return std::isnan(value) ? 0. : value; })
                    .matrix();

            forwardDistance = 1. / inverseDistances.maxCoeff();
            backwardDistance = 1. / inverseDistances.minCoeff();
            assert(backwardDistance < 0 && forwardDistance > 0);
            assert(((b - A * state).array() >= 0).all());

            step = chordStepDistribution.draw(rng, backwardDistance, forwardDistance);

            proposal(i) += step;
            proposalSlacks.noalias() -= A.col(i) * step;

            if constexpr (IsSetStepSizeAvailable<ChordStepDistribution>::value) {
                double stepSize = chordStepDistribution.getStepSize();
                double detailedBalanceState = chordStepDistribution.computeInverseNormalizationConstant(stepSize,
                                                                                                        backwardDistance,
                                                                                                        forwardDistance);
                double detailedBalanceProposal = chordStepDistribution.computeInverseNormalizationConstant(stepSize,
                                                                                                           backwardDistance -
                                                                                                           step,
                                                                                                           forwardDistance -
                                                                                                           step);
                detailedBalance += detailedBalanceState - detailedBalanceProposal;
            }

        }


        return proposal;
    }

    template<typename InternalMatrixType, typename InternalVectorType, typename ChordStepDistribution>
    VectorType &
    CoordinateHitAndRunProposal<InternalMatrixType, InternalVectorType, ChordStepDistribution>::acceptProposal() {
        if (shouldRecomputeSlacks) {
            state = proposal;
            slacks = proposalSlacks;
            detailedBalance = 0;
            return state;
        }

        state(coordinateToUpdate) += step;
        slacks.noalias() -= A.col(coordinateToUpdate) * step;
        return state;
    }

    template<typename InternalMatrixType, typename InternalVectorType, typename ChordStepDistribution>
    void CoordinateHitAndRunProposal<InternalMatrixType, InternalVectorType, ChordStepDistribution>::setState(
            const VectorType &newState) {
        if (((b - A * newState).array() < 0).any()) {
            throw std::invalid_argument("Starting point outside polytope always gives constant Markov chain.");
        }
        CoordinateHitAndRunProposal::state = newState;
        CoordinateHitAndRunProposal::proposal = CoordinateHitAndRunProposal::state;
        slacks = b - A * CoordinateHitAndRunProposal::state;
    }

    template<typename InternalMatrixType, typename InternalVectorType, typename ChordStepDistribution>
    void CoordinateHitAndRunProposal<InternalMatrixType, InternalVectorType, ChordStepDistribution>::setProposal(
            const VectorType &newProposal) {
        shouldRecomputeSlacks = true;
        CoordinateHitAndRunProposal::proposal = newProposal;
        proposalSlacks = b - A * CoordinateHitAndRunProposal::proposal;
        detailedBalance = 0;
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
        } else {
            (void) stepSize;
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
        if constexpr (IsSetStepSizeAvailable<ChordStepDistribution>::value) {
            if (shouldRecomputeSlacks) {
                return detailedBalance;
            }

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
            return 0;
        }
    }

    template<typename InternalMatrixType, typename InternalVectorType, typename ChordStepDistribution>
    bool
    CoordinateHitAndRunProposal<InternalMatrixType, InternalVectorType, ChordStepDistribution>::hasStepSize() {
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
        if (this->getStepSize().has_value()) {
            return {"step_size"};
        } else {
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
    const MatrixType &
    CoordinateHitAndRunProposal<InternalMatrixType, InternalVectorType, ChordStepDistribution>::getA() const {
        return A;
    }

    template<typename InternalMatrixType, typename InternalVectorType, typename ChordStepDistribution>
    const VectorType &
    CoordinateHitAndRunProposal<InternalMatrixType, InternalVectorType, ChordStepDistribution>::getB() const {
        return b;
    }

    template<typename InternalMatrixType, typename InternalVectorType, typename ChordStepDistribution>
    bool
    CoordinateHitAndRunProposal<InternalMatrixType, InternalVectorType, ChordStepDistribution>::isSymmetric() const {
        // As soon as there is a step size the polytope borders and normalization will make the proposal asymmetric.
        if (getStepSize()) {
            return false;
        }
        return true;
    }

    template<typename InternalMatrixType, typename InternalVectorType, typename ChordStepDistribution>
    void CoordinateHitAndRunProposal<InternalMatrixType, InternalVectorType, ChordStepDistribution>::setDimensionNames(
            const std::vector<std::string> &names) {
        dimensionNames = names;
    }

    template<typename InternalMatrixType, typename InternalVectorType, typename ChordStepDistribution>
    std::vector<std::string>
    CoordinateHitAndRunProposal<InternalMatrixType, InternalVectorType, ChordStepDistribution>::getDimensionNames() const {
        return dimensionNames;
    }
}

#endif //HOPS_COORDINATEHITANDRUNPROPOSAL_HPP
