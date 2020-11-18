#ifndef HOPS_CSMMALANOGRADIENTPROPOSAL_HPP
#define HOPS_CSMMALANOGRADIENTPROPOSAL_HPP

#include <hops/MarkovChain/Proposal/DikinProposal.hpp>
#include <hops/MarkovChain/Recorder/IsStoreMetropolisHastingsInfoRecordAvailable.hpp>
#include <hops/MarkovChain/Recorder/IsAddMessageAvailabe.hpp>
#include <random>

namespace hops {
    template<typename Model, typename Matrix>
    class CSmMALANoGradientProposal : public Model {
    public:
        using MatrixType = Matrix;
        using VectorType = typename Model::VectorType;
        using StateType = typename Model::VectorType;

        /**
         * @brief Constructs proposal mechanism on polytope defined as Ax<b.
         * @param A
         * @param b
         * @param currentState
         */
        CSmMALANoGradientProposal(const Model &model, MatrixType A, VectorType b, VectorType currentState);

        void propose(RandomNumberGenerator &randomNumberGenerator);

        void acceptProposal();

        [[nodiscard]] typename MatrixType::Scalar calculateLogAcceptanceProbability();

        StateType getState() const;

        void setState(StateType newState);

        StateType getProposal() const;

        typename MatrixType::Scalar getStepSize() const;

        void setStepSize(typename MatrixType::Scalar newStepSize);

        double getNegativeLogLikelihoodOfCurrentState();

        std::string getName();

    private:
        MatrixType A;
        VectorType b;

        StateType state;
        StateType proposal;
        typename MatrixType::Scalar stateLogSqrtDeterminant = 0;
        typename MatrixType::Scalar proposalLogSqrtDeterminant = 0;
        typename MatrixType::Scalar stateNegativeLogLikelihood = 0;
        typename MatrixType::Scalar proposalNegativeLogLikelihood = 0;
        Eigen::Matrix<typename MatrixType::Scalar, Eigen::Dynamic, Eigen::Dynamic> stateSqrtInvMetric;
        Eigen::Matrix<typename MatrixType::Scalar, Eigen::Dynamic, Eigen::Dynamic> stateMetric;
        Eigen::Matrix<typename MatrixType::Scalar, Eigen::Dynamic, Eigen::Dynamic> proposalSqrtInvMetric;
        Eigen::Matrix<typename MatrixType::Scalar, Eigen::Dynamic, Eigen::Dynamic> proposalMetric;

        typename MatrixType::Scalar stepSize;
        static constexpr const typename MatrixType::Scalar fisherWeight = 0.5;
        typename MatrixType::Scalar geometricFactor;
        typename MatrixType::Scalar covarianceFactor;

        std::normal_distribution<typename MatrixType::Scalar> normalDistribution{0., 1.};
        DikinEllipsoidCalculator<MatrixType, VectorType> dikinEllipsoidCalculator;
    };

    namespace {
        template<typename MatrixType>
        void calculateMetricInfoForCSmMALANoGradient(const MatrixType &metric,
                                 MatrixType &sqrtInvMetric,
                                 double &logSqrtDeterminant) {
            Eigen::SelfAdjointEigenSolver<MatrixType> solver(metric);
            if (solver.info() != Eigen::Success) {
                throw std::runtime_error("Decomposition failed.");
            }
            sqrtInvMetric = solver.operatorInverseSqrt();
            logSqrtDeterminant = 0.5 * solver.eigenvalues().array().log().sum();
        }
    }

    template<typename Model, typename Matrix>
    CSmMALANoGradientProposal<Model, Matrix>::CSmMALANoGradientProposal(const Model &model,
                                                                        MatrixType A,
                                                                        VectorType b,
                                                                        VectorType currentState) :
            Model(model),
            A(std::move(A)),
            b(std::move(b)),
            dikinEllipsoidCalculator(this->A, this->b) {
        setState(currentState);
        setStepSize(1.);
        proposal = state;
    }

    template<typename Model, typename Matrix>
    void CSmMALANoGradientProposal<Model, Matrix>::propose(RandomNumberGenerator &randomNumberGenerator) {
        for (long i = 0; i < proposal.rows(); ++i) {
            proposal(i) = normalDistribution(randomNumberGenerator);
        }
        proposal = state + covarianceFactor * (stateSqrtInvMetric * proposal);
    }

    template<typename Model, typename Matrix>
    void CSmMALANoGradientProposal<Model, Matrix>::acceptProposal() {
        state.swap(proposal);
        if (((A * state - b).array() > 0).any()) {
            throw std::runtime_error("Current state is outside of polytope!");
        }
        stateSqrtInvMetric.swap(proposalSqrtInvMetric);
        stateMetric.swap(proposalMetric);
        stateLogSqrtDeterminant = proposalLogSqrtDeterminant;
        stateNegativeLogLikelihood = proposalNegativeLogLikelihood;
    }

    template<typename Model, typename Matrix>
    typename CSmMALANoGradientProposal<Model, Matrix>::MatrixType::Scalar
    CSmMALANoGradientProposal<Model, Matrix>::calculateLogAcceptanceProbability() {
        bool isProposalInteriorPoint = ((A * proposal - b).array() < 0).all();
        if (!isProposalInteriorPoint) {
            if constexpr(IsStoreMetropolisHastingsInfoRecordAvailable<Model>::value) {
                Model::storeMetropolisHastingsInfoRecord("polytope");
            }
            return -std::numeric_limits<typename MatrixType::Scalar>::infinity();
        }
        if constexpr(IsStoreMetropolisHastingsInfoRecordAvailable<Model>::value) {
            Model::storeMetropolisHastingsInfoRecord("likelihood");
        }

        auto dikinEllipsoid = dikinEllipsoidCalculator.calculateDikinEllipsoid(proposal);
        auto fisherInformation = Model::calculateExpectedFisherInformation(proposal);
        proposalMetric = fisherWeight * fisherInformation + (1 - fisherWeight) * dikinEllipsoid;
        calculateMetricInfoForCSmMALANoGradient(proposalMetric, proposalSqrtInvMetric, proposalLogSqrtDeterminant);
        proposalNegativeLogLikelihood = Model::calculateNegativeLogLikelihood(proposal);

        VectorType stateDifference = state - proposal;
        double normDifference = stateDifference.transpose() * (stateMetric - proposalMetric) * stateDifference;

        return -proposalNegativeLogLikelihood
               + stateNegativeLogLikelihood
               + proposalLogSqrtDeterminant
               - stateLogSqrtDeterminant
               + geometricFactor * normDifference;
    }

    template<typename Model, typename Matrix>
    typename CSmMALANoGradientProposal<Model, Matrix>::StateType
    CSmMALANoGradientProposal<Model, Matrix>::getState() const {
        return state;
    }

    template<typename Model, typename Matrix>
    void CSmMALANoGradientProposal<Model, Matrix>::setState(StateType newState) {
        state.swap(newState);
        auto dikinEllipsoid = dikinEllipsoidCalculator.calculateDikinEllipsoid(state);
        auto fisherInformation = Model::calculateExpectedFisherInformation(state);
        stateMetric = fisherWeight * fisherInformation + (1 - fisherWeight) * dikinEllipsoid;
        calculateMetricInfoForCSmMALANoGradient(stateMetric, stateSqrtInvMetric, stateLogSqrtDeterminant);
        stateNegativeLogLikelihood = Model::calculateNegativeLogLikelihood(state);
    }

    template<typename Model, typename Matrix>
    typename CSmMALANoGradientProposal<Model, Matrix>::StateType
    CSmMALANoGradientProposal<Model, Matrix>::getProposal() const {
        return proposal;
    }

    template<typename Model, typename Matrix>
    typename CSmMALANoGradientProposal<Model, Matrix>::MatrixType::Scalar
    CSmMALANoGradientProposal<Model, Matrix>::getStepSize() const {
        return stepSize;
    }

    template<typename Model, typename Matrix>
    void CSmMALANoGradientProposal<Model, Matrix>::setStepSize(typename MatrixType::Scalar newStepSize) {
        stepSize = newStepSize;
        geometricFactor = A.cols() / (2 * stepSize * stepSize);
        covarianceFactor = stepSize / std::sqrt(A.cols());
    }

    template<typename Model, typename Matrix>
    double CSmMALANoGradientProposal<Model, Matrix>::getNegativeLogLikelihoodOfCurrentState() {
        return stateNegativeLogLikelihood;
    }

    template<typename Model, typename Matrix>
    std::string CSmMALANoGradientProposal<Model, Matrix>::getName() {
        return "CSmMALANoGradient";
    }
}

#endif //HOPS_CSMMALANOGRADIENTPROPOSAL_HPP
