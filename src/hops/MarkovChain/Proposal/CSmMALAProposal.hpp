#ifndef HOPS_CSMMALA_HPP
#define HOPS_CSMMALA_HPP

#include <Eigen/Eigenvalues>
#include <random>
#include <utility>

#include <hops/MarkovChain/Recorder/IsAddMessageAvailabe.hpp>
#include <hops/Model/Model.hpp>
#include <hops/Utility/MatrixType.hpp>
#include <hops/Utility/StringUtility.hpp>
#include <hops/Utility/VectorType.hpp>

#include "DikinEllipsoidCalculator.hpp"
#include "Proposal.hpp"

namespace hops {
    namespace CSmMALAProposalDetails {
        template<typename MatrixType>
        void computeMetricInfoForCSmMALAWithSvd(const MatrixType &metric,
                                                MatrixType &sqrtInvMetric,
                                                double &logSqrtDeterminant) {
            Eigen::BDCSVD<MatrixType> solver(metric, Eigen::ComputeFullU);
            sqrtInvMetric = solver.matrixU() * solver.singularValues().cwiseInverse().cwiseSqrt().asDiagonal() *
                            solver.matrixU().adjoint();
            logSqrtDeterminant = 0.5 * solver.singularValues().array().log().sum();
        }
    }

    /**
     * @tparam ModelType
     * @tparam InternalMatrixType
     */
    template<typename ModelType, typename InternalMatrixType>
    class CSmMALAProposal : public Proposal, public ModelType {
    public:
        /**
         * @brief Constructs proposal mechanism on polytope defined as Ax<b.
         * @param A
         * @param b
         * @param currentState
         * @param fisherWeight parameterizes the mixing of Dikin metric and Fisher information.
         */
        CSmMALAProposal(InternalMatrixType A,
                        VectorType b,
                        const VectorType &currentState,
                        ModelType model,
                        double newFisherWeight = 0.5,
                        double newStepSize = 1);

        VectorType &propose(RandomNumberGenerator &rng) override;

        VectorType &acceptProposal() override;

        void setState(const VectorType &state) override;

        [[nodiscard]] VectorType getState() const override;

        [[nodiscard]] VectorType getProposal() const override;

        [[nodiscard]] std::vector<std::string> getDimensionNames() const override;

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

        [[nodiscard]] const MatrixType& getA() const override;

        [[nodiscard]] const VectorType& getB() const override;

        [[nodiscard]] std::unique_ptr<Model> getModel() const;

        ProposalStatistics & getProposalStatistics() override;

        void activateTrackingOfProposalStatistics() override;

        void disableTrackingOfProposalStatistics() override;

        bool isTrackingOfProposalStatisticsActivated() override;

        ProposalStatistics getAndResetProposalStatistics() override;


    private:
        VectorType computeTruncatedGradient(VectorType x);

        InternalMatrixType A;
        MatrixType Adense;
        VectorType b;
        ProposalStatistics proposalStatistics;

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
        double fisherWeight = .5;
        double fisherScale = 1.;
        double geometricFactor = 0;
        double covarianceFactor = 0;

        std::normal_distribution<double> normalDistribution{0., 1.};
        DikinEllipsoidCalculator<MatrixType, VectorType> dikinEllipsoidCalculator;

        bool isProposalInfosTrackingActive = false;
    };

    template<typename ModelType, typename InternalMatrixType>
    CSmMALAProposal<ModelType, InternalMatrixType>::CSmMALAProposal(InternalMatrixType A,
                                                                    hops::VectorType b,
                                                                    const VectorType &currentState,
                                                                    ModelType model,
                                                                    double newFisherWeight,
                                                                    double newStepSize) :
            ModelType(std::move(model)),
            A(std::move(A)),
            Adense(this->A),
            b(std::move(b)),
            dikinEllipsoidCalculator(this->A, this->b) {
        if (newFisherWeight > 1 || newFisherWeight < 0) {
            throw std::invalid_argument("fisherWeight should be in [0, 1].");
        }

        this->fisherWeight = newFisherWeight;

        stateMetric = MatrixType::Zero(
                currentState.rows(), currentState.rows());
        proposalMetric = MatrixType::Zero(
                currentState.rows(), currentState.rows());
        CSmMALAProposal::setState(currentState);
        CSmMALAProposal::setStepSize(newStepSize);
        proposal = state;
    }

    template<typename ModelType, typename InternalMatrixType>
    VectorType &CSmMALAProposal<ModelType, InternalMatrixType>::propose(RandomNumberGenerator &rng) {
        for (long i = 0; i < proposal.rows(); ++i) {
            proposal(i) = normalDistribution(rng);
        }
        proposal = driftedState + covarianceFactor * (stateSqrtInvMetric * proposal);

        return proposal;
    }

    template<typename ModelType, typename InternalMatrixType>
    VectorType &CSmMALAProposal<ModelType, InternalMatrixType>::acceptProposal() {
        state.swap(proposal);
        driftedState.swap(driftedProposal);
        stateSqrtInvMetric.swap(proposalSqrtInvMetric);
        stateMetric.swap(proposalMetric);
        stateLogSqrtDeterminant = proposalLogSqrtDeterminant;
        stateNegativeLogLikelihood = proposalNegativeLogLikelihood;
        return state;
    }

    template<typename ModelType, typename InternalMatrixType>
    void CSmMALAProposal<ModelType, InternalMatrixType>::setState(const VectorType &newState) {
        if (((b - A * newState).array() < 0).any()) {
            throw std::invalid_argument("Starting point outside polytope always gives constant Markov chain.");
        }

        state = newState;

        // Important: compute gradient before fisher info or else 13CFLUX2 will throw, since it uses internal
        // gradient data to construct fisher information.
        VectorType gradient = computeTruncatedGradient(state);
        stateMetric.setZero();
        if (fisherWeight != 0) {
            std::optional<decltype(stateMetric)> optionalFisherInformation = ModelType::computeExpectedFisherInformation(
                    state);
            if (optionalFisherInformation) {
                decltype(stateMetric) fisherInformation = optionalFisherInformation.value();
                stateMetric += fisherWeight * fisherScale * fisherInformation;
            }
        }
        if (fisherWeight != 1) {
            decltype(stateMetric) dikinEllipsoid = dikinEllipsoidCalculator.computeDikinEllipsoid(state);
            stateMetric += (1 - fisherWeight) * dikinEllipsoid;
        }
        CSmMALAProposalDetails::computeMetricInfoForCSmMALAWithSvd(stateMetric,
                                                                   stateSqrtInvMetric,
                                                                   stateLogSqrtDeterminant);
        driftedState = state + 0.5 * std::pow(covarianceFactor, 2) * stateSqrtInvMetric * stateSqrtInvMetric *
                               gradient;
        stateNegativeLogLikelihood = ModelType::computeNegativeLogLikelihood(state);
    }

    template<typename ModelType, typename InternalMatrixType>
    std::optional<double> CSmMALAProposal<ModelType, InternalMatrixType>::getStepSize() const {
        return stepSize;
    }

    template<typename ModelType, typename InternalMatrixType>
    void CSmMALAProposal<ModelType, InternalMatrixType>::setStepSize(double newStepSize) {
        stepSize = newStepSize;
        geometricFactor = A.cols() / (2 * stepSize * stepSize);
        covarianceFactor = stepSize / std::sqrt(A.cols());
        setState(state);
    }

    template<typename ModelType, typename InternalMatrixType>
    std::string CSmMALAProposal<ModelType, InternalMatrixType>::getProposalName() const {
        return "CSmMALA";
    }

    template<typename ModelType, typename InternalMatrixType>
    double CSmMALAProposal<ModelType, InternalMatrixType>::getStateNegativeLogLikelihood() const {
        return stateNegativeLogLikelihood;
    }

    template<typename ModelType, typename InternalMatrixType>
    double CSmMALAProposal<ModelType, InternalMatrixType>::computeLogAcceptanceProbability() {
        bool isProposalInteriorPoint = ((A * proposal - b).array() < 0).all();
        if (!isProposalInteriorPoint) {
            if(isProposalInfosTrackingActive) {
                proposalStatistics.appendInfo("proposal_is_interior", isProposalInteriorPoint);
                proposalStatistics.appendInfo("proposal_neg_like", std::numeric_limits<double>::quiet_NaN());
            }
            return -std::numeric_limits<double>::infinity();
        }

        // Important: compute gradient before fisher info or else x3cflux2 will throw
        VectorType gradient = computeTruncatedGradient(proposal);
        proposalMetric.setZero();
        if (fisherWeight != 0) {
            std::optional<decltype(proposalMetric)> optionalFisherInformation = ModelType::computeExpectedFisherInformation(
                    proposal);
            if (optionalFisherInformation) {
                decltype(proposalMetric) fisherInformation = optionalFisherInformation.value();
                proposalMetric += (fisherWeight * fisherScale * fisherInformation);
            }
        }
        if (fisherWeight != 1) {
            decltype(proposalMetric) dikinEllipsoid = dikinEllipsoidCalculator.computeDikinEllipsoid(proposal);
            proposalMetric += (1 - fisherWeight) * dikinEllipsoid;

        }
        CSmMALAProposalDetails::computeMetricInfoForCSmMALAWithSvd(proposalMetric, proposalSqrtInvMetric,
                                                                   proposalLogSqrtDeterminant);
        driftedProposal = proposal +
                          0.5 * std::pow(covarianceFactor, 2) * proposalSqrtInvMetric * proposalSqrtInvMetric *
                          gradient;
        proposalNegativeLogLikelihood = ModelType::computeNegativeLogLikelihood(proposal);

        double normDifference =
                static_cast<double>((driftedState - proposal).transpose() * stateMetric * (driftedState - proposal)) -
                static_cast<double>((state - driftedProposal).transpose() * proposalMetric * (state - driftedProposal));

        if(isProposalInfosTrackingActive) {
            proposalStatistics.appendInfo("proposal_is_interior", isProposalInteriorPoint);
            proposalStatistics.appendInfo("proposal_neg_like", proposalNegativeLogLikelihood);
        }

        return -proposalNegativeLogLikelihood
               + stateNegativeLogLikelihood
               + proposalLogSqrtDeterminant
               - stateLogSqrtDeterminant
               + geometricFactor * normDifference;
    }

    template<typename ModelType, typename InternalMatrixType>
    VectorType CSmMALAProposal<ModelType, InternalMatrixType>::computeTruncatedGradient(VectorType x) {
        auto gradient = ModelType::computeLogLikelihoodGradient(x);
        if (gradient) {
            double norm = gradient.value().norm();
            if (norm != 0) {
                gradient.value() /= norm;
            }
            return gradient.value();
        }
        return VectorType::Zero(x.rows());
    }

    template<typename ModelType, typename InternalMatrixType>
    bool CSmMALAProposal<ModelType, InternalMatrixType>::hasStepSize() const {
        return true;
    }

    template<typename ModelType, typename InternalMatrixType>
    std::unique_ptr<Proposal> CSmMALAProposal<ModelType, InternalMatrixType>::copyProposal() const {
        return std::make_unique<CSmMALAProposal>(*this);
    }

    template<typename ModelType, typename InternalMatrixType>
    VectorType CSmMALAProposal<ModelType, InternalMatrixType>::getState() const {
        return state;
    }

    template<typename ModelType, typename InternalMatrixType>
    VectorType CSmMALAProposal<ModelType, InternalMatrixType>::getProposal() const {
        return proposal;
    }

    template<typename ModelType, typename InternalMatrixType>
    std::vector<std::string> CSmMALAProposal<ModelType, InternalMatrixType>::getParameterNames() const {
        return {"step_size", "fisher_weight"};
    }

    template<typename ModelType, typename InternalMatrixType>
    std::any CSmMALAProposal<ModelType, InternalMatrixType>::getParameter(const ProposalParameter &parameter) const {
        if (parameter == ProposalParameter::STEP_SIZE) {
            return std::any(this->stepSize);
        }
        if (parameter == ProposalParameter::FISHER_WEIGHT) {
            return std::any(this->fisherWeight);
        }
        throw std::invalid_argument("Can't get parameter which doesn't exist in " + this->getProposalName());
    }

    template<typename ModelType, typename InternalMatrixType>
    std::string
    CSmMALAProposal<ModelType, InternalMatrixType>::getParameterType(const ProposalParameter &parameter) const {
        if (parameter == ProposalParameter::STEP_SIZE) {
            return "double";
        }
        else if (parameter == ProposalParameter::FISHER_WEIGHT) {
            return "double";
        } else {
            throw std::invalid_argument("Can't get parameter which doesn't exist in " + this->getProposalName());
        }
    }

    template<typename ModelType, typename InternalMatrixType>
    void CSmMALAProposal<ModelType, InternalMatrixType>::setParameter(const ProposalParameter &parameter,
                                                                      const std::any &value) {
        if (parameter == ProposalParameter::STEP_SIZE) {
            setStepSize(std::any_cast<double>(value));
        }
        else if (parameter == ProposalParameter::FISHER_WEIGHT) {
            fisherWeight = std::any_cast<double>(value);
            // internal changes of setStepSize are a function of the value of fisherWeight, therefore recalculate here.
            setStepSize((this->stepSize));
        } else {
            throw std::invalid_argument("Can't get parameter which doesn't exist in " + this->getProposalName());
        }
    }

    template<typename ModelType, typename InternalMatrixType>
    std::vector<std::string> CSmMALAProposal<ModelType, InternalMatrixType>::getDimensionNames() const {
        return ModelType::getDimensionNames();
    }

    template<typename ModelType, typename InternalMatrixType>
    double CSmMALAProposal<ModelType, InternalMatrixType>::getProposalNegativeLogLikelihood() const {
        return proposalNegativeLogLikelihood;
    }

    template<typename ModelType, typename InternalMatrixType>
    bool CSmMALAProposal<ModelType, InternalMatrixType>::hasNegativeLogLikelihood() const {
        return true;
    }

    template<typename ModelType, typename InternalMatrixType>
    const MatrixType& CSmMALAProposal<ModelType, InternalMatrixType>::getA() const {
        return Adense;
    }

    template<typename ModelType, typename InternalMatrixType>
    const VectorType& CSmMALAProposal<ModelType, InternalMatrixType>::getB() const {
        return b;
    }

    template<typename ModelType, typename InternalMatrixType>
    std::unique_ptr<Model> CSmMALAProposal<ModelType, InternalMatrixType>::getModel() const {
        return ModelType::copyModel();
    }

    template<typename ModelType, typename InternalMatrixType>
    ProposalStatistics & CSmMALAProposal<ModelType, InternalMatrixType>::getProposalStatistics() {
        return proposalStatistics;
    }

    template<typename ModelType, typename InternalMatrixType>
    void CSmMALAProposal<ModelType, InternalMatrixType>::activateTrackingOfProposalStatistics() {
        isProposalInfosTrackingActive = true;
    }

    template<typename ModelType, typename InternalMatrixType>
    void CSmMALAProposal<ModelType, InternalMatrixType>::disableTrackingOfProposalStatistics() {
        isProposalInfosTrackingActive = false;
    }

    template<typename ModelType, typename InternalMatrixType>
    bool CSmMALAProposal<ModelType, InternalMatrixType>::isTrackingOfProposalStatisticsActivated() {
        return isProposalInfosTrackingActive;
    }

    template<typename ModelType, typename InternalMatrixType>
    ProposalStatistics CSmMALAProposal<ModelType, InternalMatrixType>::getAndResetProposalStatistics() {
        ProposalStatistics newStatistic;
        std::swap(newStatistic, proposalStatistics);
        return newStatistic;
    }
}

#endif //HOPS_CSMMALA_HPP
