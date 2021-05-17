#ifndef HOPS_CSMMALANOGRADIENTPROPOSAL_HPP
#define HOPS_CSMMALANOGRADIENTPROPOSAL_HPP

#include <Eigen/Eigenvalues>
#include "DikinProposal.hpp"
#include "../Recorder/IsStoreMetropolisHastingsInfoRecordAvailable.hpp"
#include "../Recorder/IsAddMessageAvailabe.hpp"
#include <random>

namespace hops {
    namespace CSmMALANoGradientProposalDetails {
        template<typename MatrixType>
        void calculateMetricInfoForCSmMALANoGradientWithSvd(const MatrixType &metric,
                                                            MatrixType &sqrtInvMetric,
                                                            double &logSqrtDeterminant) {
            Eigen::BDCSVD<MatrixType> solver(metric, Eigen::ComputeFullU);
            sqrtInvMetric = solver.matrixU() * solver.singularValues().cwiseInverse().cwiseSqrt().asDiagonal() *
                            solver.matrixU().adjoint();
            logSqrtDeterminant = 0.5 * solver.singularValues().array().log().sum();
        }
    }

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

        void setFisherWeight(typename MatrixType::Scalar newFisherWeight);

        double getNegativeLogLikelihoodOfCurrentState();

        std::string getName();

    private:
        MatrixType A;
        VectorType b;

        StateType state;
        StateType driftedState;
        StateType proposal;
        StateType driftedProposal;
        typename MatrixType::Scalar stateLogSqrtDeterminant = 0;
        typename MatrixType::Scalar proposalLogSqrtDeterminant = 0;
        typename MatrixType::Scalar stateNegativeLogLikelihood = 0;
        typename MatrixType::Scalar proposalNegativeLogLikelihood = 0;
        Eigen::Matrix<typename MatrixType::Scalar, Eigen::Dynamic, Eigen::Dynamic> stateSqrtInvMetric;
        Eigen::Matrix<typename MatrixType::Scalar, Eigen::Dynamic, Eigen::Dynamic> stateMetric;
        Eigen::Matrix<typename MatrixType::Scalar, Eigen::Dynamic, Eigen::Dynamic> proposalSqrtInvMetric;
        Eigen::Matrix<typename MatrixType::Scalar, Eigen::Dynamic, Eigen::Dynamic> proposalMetric;

        typename MatrixType::Scalar stepSize;
        typename MatrixType::Scalar fisherWeight = .5;
        typename MatrixType::Scalar geometricFactor;
        typename MatrixType::Scalar covarianceFactor;

        std::normal_distribution<typename MatrixType::Scalar> normalDistribution{0., 1.};
        DikinEllipsoidCalculator<MatrixType, VectorType> dikinEllipsoidCalculator;
    };

    template<typename Model,
            typename Matrix>

    CSmMALANoGradientProposal<Model, Matrix>::CSmMALANoGradientProposal(const Model &model,
                                                                        MatrixType A,
                                                                        VectorType b,
                                                                        VectorType currentState) :
            Model(model),
            A(std::move(A)),
            b(std::move(b)),
            dikinEllipsoidCalculator(this->A, this->b) {
        stateMetric = Eigen::Matrix<typename MatrixType::Scalar, Eigen::Dynamic, Eigen::Dynamic>::Zero(
                currentState.rows(), currentState.rows());
        proposalMetric = Eigen::Matrix<typename MatrixType::Scalar, Eigen::Dynamic, Eigen::Dynamic>::Zero(
                currentState.rows(), currentState.rows());
        setState(currentState);
        setStepSize(1.);
        proposal = state;
    }

    template<typename Model,
            typename Matrix>

    void CSmMALANoGradientProposal<Model, Matrix>::propose(
            RandomNumberGenerator &randomNumberGenerator) {
        for (long i = 0; i < proposal.rows(); ++i) {
            proposal(i) = normalDistribution(randomNumberGenerator);
        }
        proposal = driftedState + covarianceFactor * (stateSqrtInvMetric * proposal);
    }

    template<typename Model,
            typename Matrix>

    void CSmMALANoGradientProposal<Model, Matrix>::acceptProposal() {
        state.swap(proposal);
        driftedState.swap(driftedProposal);
        stateSqrtInvMetric.swap(proposalSqrtInvMetric);
        stateMetric.swap(proposalMetric);
        stateLogSqrtDeterminant = proposalLogSqrtDeterminant;
        stateNegativeLogLikelihood = proposalNegativeLogLikelihood;
    }

    template<typename Model,
            typename Matrix>

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

        proposalMetric.setZero();
        if (fisherWeight != 0) {
            // Important: calculate gradient before fisher info or else x3cflux2 will throw
            StateType gradient = Model::calculateLogLikelihoodGradient(proposal);
            auto fisherInformation = Model::calculateExpectedFisherInformation(proposal);
            proposalMetric += fisherWeight * fisherInformation;
        }
        if (fisherWeight != 1) {
            auto dikinEllipsoid = dikinEllipsoidCalculator.calculateDikinEllipsoid(proposal);
            proposalMetric += (1 - fisherWeight) * dikinEllipsoid;

        }
        CSmMALANoGradientProposalDetails::calculateMetricInfoForCSmMALANoGradientWithSvd(proposalMetric, proposalSqrtInvMetric, proposalLogSqrtDeterminant);
        driftedProposal = proposal;
        proposalNegativeLogLikelihood = Model::calculateNegativeLogLikelihood(proposal);

        double normDifference =
                static_cast<double>((driftedState - proposal).transpose() * stateMetric * (driftedState - proposal)) -
                static_cast<double>((state - driftedProposal).transpose() * proposalMetric * (state - driftedProposal));

        return -proposalNegativeLogLikelihood
               + stateNegativeLogLikelihood
               + proposalLogSqrtDeterminant
               - stateLogSqrtDeterminant
               + geometricFactor * normDifference;
    }

    template<typename Model,
            typename Matrix>

    typename CSmMALANoGradientProposal<Model, Matrix>::StateType
    CSmMALANoGradientProposal<Model, Matrix>::getState() const {
        return state;
    }

    template<typename Model,
            typename Matrix>

    void CSmMALANoGradientProposal<Model, Matrix>::setState(StateType newState) {
        state.swap(newState);
        stateMetric.setZero();
        if (fisherWeight != 0) {
            // Important: calculate gradient before fisher info or else x3cflux2 will throw
            StateType gradient = Model::calculateLogLikelihoodGradient(state);
            auto fisherInformation = Model::calculateExpectedFisherInformation(state);
            stateMetric += fisherWeight * fisherInformation;
        }
        if (fisherWeight != 1) {
            auto dikinEllipsoid = dikinEllipsoidCalculator.calculateDikinEllipsoid(state);
            stateMetric += (1 - fisherWeight) * dikinEllipsoid;
        }
        CSmMALANoGradientProposalDetails::calculateMetricInfoForCSmMALANoGradientWithSvd(stateMetric, stateSqrtInvMetric, stateLogSqrtDeterminant);
        driftedState = state;
        stateNegativeLogLikelihood = Model::calculateNegativeLogLikelihood(state);
    }

    template<typename Model,
            typename Matrix>

    typename CSmMALANoGradientProposal<Model, Matrix>::StateType
    CSmMALANoGradientProposal<Model, Matrix>::getProposal() const {
        return proposal;
    }

    template<typename Model,
            typename Matrix>

    typename CSmMALANoGradientProposal<Model, Matrix>::MatrixType::Scalar
    CSmMALANoGradientProposal<Model, Matrix>::getStepSize() const {
        return stepSize;
    }

    template<typename Model,
            typename Matrix>

    void
    CSmMALANoGradientProposal<Model, Matrix>::setStepSize(typename MatrixType::Scalar newStepSize) {
        stepSize = newStepSize;
        geometricFactor = A.cols() / (2 * stepSize * stepSize);
        covarianceFactor = stepSize / std::sqrt(A.cols());
        setState(state);
    }

    template<typename Model,
            typename Matrix>

    void
    CSmMALANoGradientProposal<Model, Matrix>::setFisherWeight(typename MatrixType::Scalar newFisherWeight) {
        if (fisherWeight > 1 || fisherWeight < 0) {
            throw std::runtime_error("fisherWeigth should be in [0, 1].");
        }
        fisherWeight = newFisherWeight;
        setState(state);
    }

    template<typename Model,
            typename Matrix>

    double CSmMALANoGradientProposal<Model, Matrix>::getNegativeLogLikelihoodOfCurrentState() {
        return stateNegativeLogLikelihood;
    }

    template<typename Model,
            typename Matrix>

    std::string CSmMALANoGradientProposal<Model, Matrix>::getName() {
        return "CSmMALANoGradient";
    }
}

#endif //HOPS_CSMMALANOGRADIENTPROPOSAL_HPP
