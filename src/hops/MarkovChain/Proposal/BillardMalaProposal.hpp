#ifndef HOPS_REFLECTIVEMALA_HPP
#define HOPS_REFLECTIVEMALA_HPP

#include <Eigen/Eigenvalues>
#include <random>
#include <utility>

#include <hops/MarkovChain/Recorder/IsAddMessageAvailabe.hpp>
#include <hops/Utility/MatrixType.hpp>
#include <hops/Utility/StringUtility.hpp>
#include <hops/Utility/VectorType.hpp>

#include "Proposal.hpp"

namespace hops {
    namespace BillardMALAProposalDetails {
        void computeMetricInfoForReflectiveMALAWithSvd(const MatrixType &metric,
                                                       MatrixType &sqrtInvMetric,
                                                       double &logSqrtDeterminant) {
            Eigen::BDCSVD<MatrixType> solver(metric, Eigen::ComputeFullU);
            sqrtInvMetric = solver.matrixU() * solver.singularValues().cwiseInverse().cwiseSqrt().asDiagonal() *
                            solver.matrixU().adjoint();
            logSqrtDeterminant = 0.5 * solver.singularValues().array().log().sum();
        }

        /**
         * @brief Assumes startPoint is interior point and endPoint is not.
         * @param A
         * @param b
         * @param startPointBillardMALAProposalDetails {
        void computeMetricInfoForReflectiveMALAWithSvd(const MatrixType &metric,
                                                       MatrixType &sqrtInvMetric
         * @param endPoint the endPoint is reflected into the polytope
         * @return number of reflections and endpoint which has been reflected into polytope
         */
        std::pair<long, VectorType> reflectIntoPolytope(const MatrixType &A,
                                       const MatrixType &b,
                                       const VectorType &startPoint,
                                       const VectorType &endPoint,
                                       long maximumNumberOfReflections
                                       ) {
            double epsilon = 1e-14;
            VectorType currentPoint = startPoint;
            VectorType trajectory = endPoint - startPoint;

            VectorType::Scalar trajectoryLength = trajectory.norm();
            VectorType trajectoryDirection = trajectory / trajectoryLength;

            VectorType slacks = b - A * startPoint;

            long numberOfReflections = 0;
            do {
                double distanceToBorder = 1. / ((A * trajectoryDirection).cwiseQuotient(slacks)).maxCoeff();
                // TODO by storing the cardinality and indices of max-coeff here the later reflection step could be sped up
                if (trajectoryLength < distanceToBorder) {
                    currentPoint += trajectoryDirection * trajectoryLength;
                    trajectoryLength = 0; // No remaining trajectoryLength to traverse
                } else {
                    numberOfReflections++;
                    trajectoryLength -= distanceToBorder;
                    currentPoint += trajectoryDirection * distanceToBorder;
                    slacks.noalias() -= A * trajectoryDirection * distanceToBorder;
                    for (int i = 0; i < A.rows(); ++i) {
                        if (slacks[i] < epsilon) {
                            trajectoryDirection -= 2 * A.row(i) * (A.row(i).transpose() * trajectoryDirection) /
                                                   A.row(i).squaredNorm();
                        }
                    }
                }
            } while (trajectoryLength > 0 && numberOfReflections < maximumNumberOfReflections);

            if (numberOfReflections < maximumNumberOfReflections) {
                return std::make_pair(numberOfReflections, currentPoint);
            }
            // Returns endpoint which is not in polytope and which will therefore be rejected.
            return std::make_pair(numberOfReflections, endPoint);
        }
    }

    /**
     * @Brief Does not work in with DNest4 sampler, because this Proposal already contains the model likelihood.
     * @tparam ModelType
     * @tparam InternalMatrixType
     */
    template<typename ModelType, typename InternalMatrixType>
    class BillardMalaProposal : public Proposal, public ModelType {
    public:
        /**
         * @brief Constructs proposal mechanism on polytope defined as Ax<b.
         * @param A
         * @param b
         * @param currentState
         */
        BillardMalaProposal(InternalMatrixType A,
                            VectorType b,
                            const VectorType &currentState,
                            ModelType model,
                            double newStepSize = 1);

        VectorType &propose(RandomNumberGenerator &rng) override;

        VectorType &acceptProposal() override;

        void setState(const VectorType &state) override;

        [[nodiscard]] VectorType getState() const override;

        [[nodiscard]] VectorType getProposal() const override;

        [[nodiscard]] std::optional<std::vector<std::string>> getDimensionNames() const override;

        [[nodiscard]] std::optional<double> getStepSize() const;

        void setStepSize(double stepSize);

        [[nodiscard]] bool hasStepSize() const override;

        [[nodiscard]] std::vector<std::string> getParameterNames() const override;

        [[nodiscard]] std::any getParameter(const ProposalParameter &parameter) const override;

        [[nodiscard]] std::string getParameterType(const ProposalParameter &parameter) const override;

        void setParameter(const ProposalParameter &parameter, const std::any &value) override;

        [[nodiscard]] std::string getProposalName() const override;

        [[nodiscard]] double getStateNegativeLogLikelihood() const override;

        [[nodiscard]] double getProposalNegativeLogLikelihood() const override;

        [[nodiscard]] bool hasNegativeLogLikelihood() const override;

        [[nodiscard]] std::unique_ptr<Proposal> copyProposal() const override;

        [[nodiscard]] double computeLogAcceptanceProbability() override;

    private:
        VectorType computeGradient(VectorType x);

        InternalMatrixType A;
        VectorType b;

        VectorType state;
        VectorType driftedState;
        VectorType proposal;
        VectorType driftedProposal;

        double stateLogSqrtDeterminant = 0;
        double proposalLogSqrtDeterminant = 0;
        double stateNegativeLogLikelihood = 0;
        double proposalNegativeLogLikelihood = 0;

        MatrixType stateSqrtInvMetric;
        MatrixType stateMetric;
        MatrixType proposalSqrtInvMetric;
        MatrixType proposalMetric;

        double stepSize = 1;
        double geometricFactor = 0;
        double covarianceFactor = 0;
        long max_num_reflections;

        std::normal_distribution<double> normalDistribution{0., 1.};
    };

    template<typename ModelType, typename InternalMatrixType>
    BillardMalaProposal<ModelType, InternalMatrixType>::BillardMalaProposal(InternalMatrixType A,
                                                                            hops::VectorType b,
                                                                            const VectorType &currentState,
                                                                            ModelType model,
                                                                            double newStepSize) :
            ModelType(std::move(model)),
            A(std::move(A)),
            b(std::move(b)) {
        BillardMalaProposal::setState(currentState);
        BillardMalaProposal::setStepSize(newStepSize);

        proposal = state;
        proposalNegativeLogLikelihood = stateNegativeLogLikelihood;
        proposalMetric = stateMetric;
        proposalSqrtInvMetric = stateSqrtInvMetric;
        proposalLogSqrtDeterminant = stateLogSqrtDeterminant;
        max_num_reflections = 10*A.cols();
    }

    template<typename ModelType, typename InternalMatrixType>
    VectorType &BillardMalaProposal<ModelType, InternalMatrixType>::propose(RandomNumberGenerator &rng) {
        for (long i = 0; i < proposal.rows(); ++i) {
            proposal(i) = normalDistribution(rng);
        }
        proposal = driftedState + covarianceFactor * (stateSqrtInvMetric * proposal);

        auto [numberOfReflections, reflectedProposal] = BillardMALAProposalDetails::reflectIntoPolytope(A, b, state, proposal,
                                                                                                        max_num_reflections);
        proposal = reflectedProposal;
        return proposal;
    }

    template<typename ModelType, typename InternalMatrixType>
    VectorType &BillardMalaProposal<ModelType, InternalMatrixType>::acceptProposal() {
        state.swap(proposal);
        driftedState.swap(driftedProposal);
        stateSqrtInvMetric.swap(proposalSqrtInvMetric);
        stateMetric.swap(proposalMetric);
        stateLogSqrtDeterminant = proposalLogSqrtDeterminant;
        stateNegativeLogLikelihood = proposalNegativeLogLikelihood;
        return state;
    }

    template<typename ModelType, typename InternalMatrixType>
    void BillardMalaProposal<ModelType, InternalMatrixType>::setState(const VectorType &newState) {
        state = newState;
        // Important: compute gradient before fisher info or else 13CFLUX2 will throw, since it uses internal
        // gradient data to construct fisher information.
        VectorType gradient = computeGradient(state);

        std::optional<decltype(stateMetric)> optionalFisherInformation = ModelType::computeExpectedFisherInformation(
                state);

        if (optionalFisherInformation) {
            stateMetric = optionalFisherInformation.value();
        } else {
            stateMetric = MatrixType::Identity(state.rows(), state.rows());
        }

        BillardMALAProposalDetails::computeMetricInfoForReflectiveMALAWithSvd(stateMetric,
                                                                              stateSqrtInvMetric,
                                                                              stateLogSqrtDeterminant);
        driftedState = state + 0.5 * std::pow(covarianceFactor, 2) * stateSqrtInvMetric * stateSqrtInvMetric *
                               gradient;
        stateNegativeLogLikelihood = ModelType::computeNegativeLogLikelihood(state);
    }

    template<typename ModelType, typename InternalMatrixType>
    std::optional<double> BillardMalaProposal<ModelType, InternalMatrixType>::getStepSize() const {
        return stepSize;
    }

    template<typename ModelType, typename InternalMatrixType>
    void BillardMalaProposal<ModelType, InternalMatrixType>::setStepSize(double newStepSize) {
        stepSize = newStepSize;
        geometricFactor = A.cols() / (2 * stepSize * stepSize);
        covarianceFactor = stepSize / std::sqrt(A.cols());
        setState(state);
    }

    template<typename ModelType, typename InternalMatrixType>
    std::string BillardMalaProposal<ModelType, InternalMatrixType>::getProposalName() const {
        return "BillardMALA";
    }

    template<typename ModelType, typename InternalMatrixType>
    double BillardMalaProposal<ModelType, InternalMatrixType>::getStateNegativeLogLikelihood() const {
        return stateNegativeLogLikelihood;
    }

    template<typename ModelType, typename InternalMatrixType>
    double BillardMalaProposal<ModelType, InternalMatrixType>::computeLogAcceptanceProbability() {
        bool isProposalInteriorPoint = ((A * proposal - b).array() < 0).all();
        if (!isProposalInteriorPoint) {
            return -std::numeric_limits<double>::infinity();
        }
        // Important: compute gradient before fisher info or else x3cflux2 will throw
        VectorType gradient = computeGradient(proposal);
        std::optional<decltype(proposalMetric)> optionalFisherInformation = ModelType::computeExpectedFisherInformation(
                proposal);
        if (optionalFisherInformation) {
            proposalMetric = optionalFisherInformation.value();
        } else {
            proposalMetric = MatrixType::Identity(state.rows(), state.rows());
        }

        BillardMALAProposalDetails::computeMetricInfoForReflectiveMALAWithSvd(proposalMetric, proposalSqrtInvMetric,
                                                                              proposalLogSqrtDeterminant);
        driftedProposal = proposal +
                          0.5 * std::pow(covarianceFactor, 2) * proposalSqrtInvMetric * proposalSqrtInvMetric *
                          gradient;

        proposalNegativeLogLikelihood = ModelType::computeNegativeLogLikelihood(proposal);

        double normDifference =
                static_cast<double>((driftedState - proposal).transpose() * stateMetric * (driftedState - proposal)) -
                static_cast<double>((state - driftedProposal).transpose() * proposalMetric * (state - driftedProposal));

        return -proposalNegativeLogLikelihood
        + stateNegativeLogLikelihood
        +proposalLogSqrtDeterminant
               - stateLogSqrtDeterminant
               + geometricFactor * normDifference;
    }

    template<typename ModelType, typename InternalMatrixType>
    VectorType BillardMalaProposal<ModelType, InternalMatrixType>::computeGradient(VectorType x) {
        auto gradient = ModelType::computeLogLikelihoodGradient(x);
        if (gradient) {
            return gradient.value();
        }
        return VectorType::Zero(x.rows());
    }

    template<typename ModelType, typename InternalMatrixType>
    bool BillardMalaProposal<ModelType, InternalMatrixType>::hasStepSize() const {
        return true;
    }

    template<typename ModelType, typename InternalMatrixType>
    std::unique_ptr<Proposal> BillardMalaProposal<ModelType, InternalMatrixType>::copyProposal() const {
        return std::make_unique<BillardMalaProposal>(*this);
    }

    template<typename ModelType, typename InternalMatrixType>
    VectorType BillardMalaProposal<ModelType, InternalMatrixType>::getState() const {
        return state;
    }

    template<typename ModelType, typename InternalMatrixType>
    VectorType BillardMalaProposal<ModelType, InternalMatrixType>::getProposal() const {
        return proposal;
    }

    template<typename ModelType, typename InternalMatrixType>
    std::vector<std::string> BillardMalaProposal<ModelType, InternalMatrixType>::getParameterNames() const {
        return {"step_size"};
    }

    template<typename ModelType, typename InternalMatrixType>
    std::any
    BillardMalaProposal<ModelType, InternalMatrixType>::getParameter(const ProposalParameter &parameter) const {
        if (parameter == ProposalParameter::STEP_SIZE) {
            return std::any(this->stepSize);
        }
        throw std::invalid_argument("Can't get parameter which doesn't exist in " + this->getProposalName());
    }

    template<typename ModelType, typename InternalMatrixType>
    std::string
    BillardMalaProposal<ModelType, InternalMatrixType>::getParameterType(const ProposalParameter &parameter) const {
        if (parameter == ProposalParameter::STEP_SIZE) {
            return "double";
        }
        if (parameter == ProposalParameter::FISHER_WEIGHT) {
            return "double";
        } else {
            throw std::invalid_argument("Can't get parameter which doesn't exist in " + this->getProposalName());
        }
    }

    template<typename ModelType, typename InternalMatrixType>
    void BillardMalaProposal<ModelType, InternalMatrixType>::setParameter(const ProposalParameter &parameter,
                                                                          const std::any &value) {
        if (parameter == ProposalParameter::STEP_SIZE) {
            setStepSize(std::any_cast<double>(value));
        } else {
            throw std::invalid_argument("Can't get parameter which doesn't exist in " + this->getProposalName());
        }
    }

    template<typename ModelType, typename InternalMatrixType>
    std::optional<std::vector<std::string>>
    BillardMalaProposal<ModelType, InternalMatrixType>::getDimensionNames() const {
        std::vector<std::string> names;
        for (long i = 0; i < state.rows(); ++i) {
            names.emplace_back("x_" + std::to_string(i));
        }
        return names;
    }

    template<typename ModelType, typename InternalMatrixType>
    double BillardMalaProposal<ModelType, InternalMatrixType>::getProposalNegativeLogLikelihood() const {
        return proposalNegativeLogLikelihood;
    }

    template<typename ModelType, typename InternalMatrixType>
    bool BillardMalaProposal<ModelType, InternalMatrixType>::hasNegativeLogLikelihood() const {
        return true;
    }
}

#endif //HOPS_REFLECTIVEMALA_HPP
