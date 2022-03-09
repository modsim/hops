#ifndef HOPS_BILLIARDMALA_HPP
#define HOPS_BILLIARDMALA_HPP

#include <Eigen/Eigenvalues>
#include <random>
#include <utility>

#include <hops/Model/Model.hpp>
#include <hops/MarkovChain/ParallelTempering/Coldness.hpp>
#include <hops/Utility/MatrixType.hpp>
#include <hops/Utility/StringUtility.hpp>
#include <hops/Utility/VectorType.hpp>
#include <hops/Transformation/Transformation.hpp>

#include "Proposal.hpp"
#include "Reflector.hpp"

namespace hops {
    namespace BilliardMALAProposalDetails {
        void computeMetricInfoForReflectiveMALAWithSvd(const MatrixType &metric,
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
    class BilliardMALAProposal : public Proposal, public ModelType {
    public:
        /**
         * @brief Constructs proposal mechanism on polytope defined as Ax<b.
         * @param A
         * @param b
         * @param currentState
         */
        BilliardMALAProposal(InternalMatrixType A,
                             VectorType b,
                             const VectorType &currentState,
                             ModelType model,
                             long maxReflections,
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

        ProposalStatistics &getProposalStatistics() override;

        void activateTrackingOfProposalStatistics() override;

        void disableTrackingOfProposalStatistics() override;

        bool isTrackingOfProposalStatisticsActivated() override;

        ProposalStatistics getAndResetProposalStatistics() override;

        const MatrixType &getA() const override;

        const VectorType &getB() const override;

        [[nodiscard]] std::unique_ptr<Model> getModel() const;

    private:
        VectorType computeGradient(VectorType x);

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
        MatrixType stateInvMetric;
        MatrixType stateMetric;
        MatrixType proposalSqrtInvMetric;
        MatrixType proposalInvMetric;
        MatrixType proposalMetric;

        double stepSize = 1;
        double geometricFactor = 0;
        double covarianceFactor = 0;

        std::normal_distribution<double> normalDistribution{0., 1.};

        long maxNumberOfReflections;
        bool isProposalInfosTrackingActive = false;
    };

    template<typename ModelType, typename InternalMatrixType>
    BilliardMALAProposal<ModelType, InternalMatrixType>::BilliardMALAProposal(InternalMatrixType A,
                                                                              VectorType b,
                                                                              const VectorType &currentState,
                                                                              ModelType model,
                                                                              long maxReflections,
                                                                              double newStepSize) :
            ModelType(std::move(model)),
            A(std::move(A)),
            Adense(MatrixType(this->A)),
            b(std::move(b)),
            maxNumberOfReflections(maxReflections) {
        if(ModelType::hasConstantExpectedFisherInformation()) {
            stateMetric = ModelType::computeExpectedFisherInformation(currentState).value();

            BilliardMALAProposalDetails::computeMetricInfoForReflectiveMALAWithSvd(stateMetric,
                                                                                   stateSqrtInvMetric,
                                                                                   stateLogSqrtDeterminant);
            stateInvMetric = stateSqrtInvMetric * stateSqrtInvMetric;
        }
        BilliardMALAProposal::setState(currentState);
        BilliardMALAProposal::setStepSize(newStepSize);

        proposal = state;
        proposalNegativeLogLikelihood = stateNegativeLogLikelihood;
        proposalMetric = stateMetric;
        proposalSqrtInvMetric = stateSqrtInvMetric;
        proposalInvMetric = stateInvMetric;
        proposalLogSqrtDeterminant = stateLogSqrtDeterminant;
    }

    template<typename ModelType, typename InternalMatrixType>
    VectorType &BilliardMALAProposal<ModelType, InternalMatrixType>::propose(RandomNumberGenerator &rng) {
        for (long i = 0; i < proposal.rows(); ++i) {
            proposal(i) = normalDistribution(rng);
        }
        proposal = driftedState + covarianceFactor * (stateSqrtInvMetric * proposal);

        const auto &reflectionResult = Reflector::reflectIntoPolytope(Adense, b, state, proposal,
                                                                      maxNumberOfReflections);
        if (isProposalInfosTrackingActive) {
            proposalStatistics.appendInfo("reflection_successful", std::get<0>(reflectionResult));
            proposalStatistics.appendInfo("number_of_reflections", std::get<1>(reflectionResult));
        }

        proposal = std::get<2>(reflectionResult);
        return proposal;
    }

    template<typename ModelType, typename InternalMatrixType>
    VectorType &BilliardMALAProposal<ModelType, InternalMatrixType>::acceptProposal() {
        state.swap(proposal);
        driftedState.swap(driftedProposal);
        stateNegativeLogLikelihood = proposalNegativeLogLikelihood;
        if(!ModelType::hasConstantExpectedFisherInformation()) {
            stateSqrtInvMetric.swap(proposalSqrtInvMetric);
            stateMetric.swap(proposalMetric);
            stateLogSqrtDeterminant = proposalLogSqrtDeterminant;
        }
        return state;
    }

    template<typename ModelType, typename InternalMatrixType>
    void BilliardMALAProposal<ModelType, InternalMatrixType>::setState(const VectorType &newState) {
        if (((b - A * newState).array() < 0).any()) {
            throw std::invalid_argument("Starting point outside polytope always gives constant Markov chain.");
        }

        state = newState;
        // Important: compute gradient before fisher info or else 13CFLUX2 will throw, since it uses internal
        // gradient data to construct fisher information.
        VectorType gradient = computeGradient(state);

        if(!ModelType::hasConstantExpectedFisherInformation()) {
            std::optional<decltype(stateMetric)> optionalFisherInformation = ModelType::computeExpectedFisherInformation(
                    state);

            if (optionalFisherInformation) {
                stateMetric = optionalFisherInformation.value();
            } else {
                stateMetric = MatrixType::Identity(state.rows(), state.rows());
            }

            BilliardMALAProposalDetails::computeMetricInfoForReflectiveMALAWithSvd(stateMetric,
                                                                                   stateSqrtInvMetric,
                                                                                   stateLogSqrtDeterminant);
        }

        driftedState = state + 0.5 * std::pow(covarianceFactor, 2) * stateInvMetric *
                               gradient;
        stateNegativeLogLikelihood = ModelType::computeNegativeLogLikelihood(state);
    }

    template<typename ModelType, typename InternalMatrixType>
    std::optional<double> BilliardMALAProposal<ModelType, InternalMatrixType>::getStepSize() const {
        return stepSize;
    }

    template<typename ModelType, typename InternalMatrixType>
    void BilliardMALAProposal<ModelType, InternalMatrixType>::setStepSize(double newStepSize) {
        stepSize = newStepSize;
        geometricFactor = A.cols() / (2 * stepSize * stepSize);
        covarianceFactor = stepSize / std::sqrt(A.cols());
        setState(state);
    }

    template<typename ModelType, typename InternalMatrixType>
    std::string BilliardMALAProposal<ModelType, InternalMatrixType>::getProposalName() const {
        return "BilliardMALA";
    }

    template<typename ModelType, typename InternalMatrixType>
    double BilliardMALAProposal<ModelType, InternalMatrixType>::getStateNegativeLogLikelihood() const {
        return stateNegativeLogLikelihood;
    }

    template<typename ModelType, typename InternalMatrixType>
    double BilliardMALAProposal<ModelType, InternalMatrixType>::computeLogAcceptanceProbability() {
        bool isProposalInteriorPoint = ((A * proposal - b).array() < 0).all();
        if (!isProposalInteriorPoint) {
            if (isProposalInfosTrackingActive) {
                proposalStatistics.appendInfo("proposal_is_interior", isProposalInteriorPoint);
                proposalStatistics.appendInfo("proposal_neg_like", std::numeric_limits<double>::quiet_NaN());
            }
            return -std::numeric_limits<double>::infinity();
        }
        // Important: compute gradient before fisher info or else x3cflux2 will throw
        VectorType gradient = computeGradient(proposal);

        if(!ModelType::hasConstantExpectedFisherInformation()) {
            std::optional<decltype(proposalMetric)> optionalFisherInformation = ModelType::computeExpectedFisherInformation(
                    proposal);
            if (optionalFisherInformation) {
                proposalMetric = optionalFisherInformation.value();
            } else {
                proposalMetric = MatrixType::Identity(state.rows(), state.rows());
            }

        BilliardMALAProposalDetails::computeMetricInfoForReflectiveMALAWithSvd(proposalMetric, proposalSqrtInvMetric,
                                                                               proposalLogSqrtDeterminant);
        }

        driftedProposal = proposal +
                          0.5 * std::pow(covarianceFactor, 2) * proposalInvMetric *
                          gradient;

        proposalNegativeLogLikelihood = ModelType::computeNegativeLogLikelihood(proposal);

        double normDifference =
                static_cast<double>((driftedState - proposal).transpose() * stateMetric * (driftedState - proposal)) -
                static_cast<double>((state - driftedProposal).transpose() * proposalMetric * (state - driftedProposal));

        if (isProposalInfosTrackingActive) {
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
    VectorType BilliardMALAProposal<ModelType, InternalMatrixType>::computeGradient(VectorType x) {
        auto gradient = ModelType::computeLogLikelihoodGradient(x);
        if (gradient) {
            return gradient.value();
        }
        return VectorType::Zero(x.rows());
    }

    template<typename ModelType, typename InternalMatrixType>
    bool BilliardMALAProposal<ModelType, InternalMatrixType>::hasStepSize() const {
        return true;
    }

    template<typename ModelType, typename InternalMatrixType>
    std::unique_ptr<Proposal> BilliardMALAProposal<ModelType, InternalMatrixType>::copyProposal() const {
        return std::make_unique<BilliardMALAProposal>(*this);
    }

    template<typename ModelType, typename InternalMatrixType>
    VectorType BilliardMALAProposal<ModelType, InternalMatrixType>::getState() const {
        return state;
    }

    template<typename ModelType, typename InternalMatrixType>
    VectorType BilliardMALAProposal<ModelType, InternalMatrixType>::getProposal() const {
        return proposal;
    }

    template<typename ModelType, typename InternalMatrixType>
    std::vector<std::string> BilliardMALAProposal<ModelType, InternalMatrixType>::getParameterNames() const {
        return {"step_size", "max_reflections"};
    }

    template<typename ModelType, typename InternalMatrixType>
    std::any
    BilliardMALAProposal<ModelType, InternalMatrixType>::getParameter(const ProposalParameter &parameter) const {
        if (parameter == ProposalParameter::STEP_SIZE) {
            return std::any(this->stepSize);
        }
        if (parameter == ProposalParameter::MAX_REFLECTIONS) {
            return std::any(this->maxNumberOfReflections);
        }
        throw std::invalid_argument("Can't get parameter which doesn't exist in " + this->getProposalName());
    }

    template<typename ModelType, typename InternalMatrixType>
    std::string
    BilliardMALAProposal<ModelType, InternalMatrixType>::getParameterType(const ProposalParameter &parameter) const {
        if (parameter == ProposalParameter::STEP_SIZE) {
            return "double";
        } else if (parameter == ProposalParameter::MAX_REFLECTIONS) {
            return "long";
        } else {
            throw std::invalid_argument("Can't get parameter which doesn't exist in " + this->getProposalName());
        }
    }

    template<typename ModelType, typename InternalMatrixType>
    void BilliardMALAProposal<ModelType, InternalMatrixType>::setParameter(const ProposalParameter &parameter,
                                                                           const std::any &value) {
        if (parameter == ProposalParameter::STEP_SIZE) {
            setStepSize(std::any_cast<double>(value));
        } else if (parameter == ProposalParameter::MAX_REFLECTIONS) {
            maxNumberOfReflections = std::any_cast<long>(value);
        } else {
            throw std::invalid_argument("Can't get parameter which doesn't exist in " + this->getProposalName());
        }
    }

    template<typename ModelType, typename InternalMatrixType>
    std::vector<std::string>
    BilliardMALAProposal<ModelType, InternalMatrixType>::getDimensionNames() const {
        return ModelType::getDimensionNames();
    }

    template<typename ModelType, typename InternalMatrixType>
    double BilliardMALAProposal<ModelType, InternalMatrixType>::getProposalNegativeLogLikelihood() const {
        return proposalNegativeLogLikelihood;
    }

    template<typename ModelType, typename InternalMatrixType>
    bool BilliardMALAProposal<ModelType, InternalMatrixType>::hasNegativeLogLikelihood() const {
        return true;
    }

    template<typename ModelType, typename InternalMatrixType>
    ProposalStatistics &BilliardMALAProposal<ModelType, InternalMatrixType>::getProposalStatistics() {
        return proposalStatistics;
    }

    template<typename ModelType, typename InternalMatrixType>
    void BilliardMALAProposal<ModelType, InternalMatrixType>::activateTrackingOfProposalStatistics() {
        isProposalInfosTrackingActive = true;
    }

    template<typename ModelType, typename InternalMatrixType>
    void BilliardMALAProposal<ModelType, InternalMatrixType>::disableTrackingOfProposalStatistics() {
        isProposalInfosTrackingActive = false;
    }

    template<typename ModelType, typename InternalMatrixType>
    bool BilliardMALAProposal<ModelType, InternalMatrixType>::isTrackingOfProposalStatisticsActivated() {
        return isProposalInfosTrackingActive;
    }

    template<typename ModelType, typename InternalMatrixType>
    ProposalStatistics BilliardMALAProposal<ModelType, InternalMatrixType>::getAndResetProposalStatistics() {
        ProposalStatistics newStatistic;
        std::swap(newStatistic, proposalStatistics);
        return newStatistic;
    }

    template<typename ModelType, typename InternalMatrixType>
    const MatrixType &BilliardMALAProposal<ModelType, InternalMatrixType>::getA() const {
        return Adense;
    }

    template<typename ModelType, typename InternalMatrixType>
    const VectorType &BilliardMALAProposal<ModelType, InternalMatrixType>::getB() const {
        return b;
    }

    template<typename ModelType, typename InternalMatrixType>
    std::unique_ptr<Model> BilliardMALAProposal<ModelType, InternalMatrixType>::getModel() const {
        return ModelType::copyModel();
    }
}

#endif //HOPS_BILLIARDMALA_HPP
