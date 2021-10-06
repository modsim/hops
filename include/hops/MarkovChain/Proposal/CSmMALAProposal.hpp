#ifndef HOPS_CSMMALA_HPP
#define HOPS_CSMMALA_HPP

#include <Eigen/Eigenvalues>
#include <hops/MarkovChain/Proposal/DikinProposal.hpp>
#include <hops/MarkovChain/Recorder/IsAddMessageAvailabe.hpp>
#include <random>

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

    template<typename Model, typename Matrix>
    class CSmMALAProposal : public Model {
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
        CSmMALAProposal(const Model &model, MatrixType A, VectorType b, VectorType currentState);

        void propose(RandomNumberGenerator &randomNumberGenerator);

        void acceptProposal();

        [[nodiscard]] typename MatrixType::Scalar computeLogAcceptanceProbability();

        StateType getState() const;

        void setState(StateType newState);

        StateType getProposal() const;

        typename MatrixType::Scalar getStepSize() const;

        void setStepSize(typename MatrixType::Scalar newStepSize);

        void setFisherWeight(typename MatrixType::Scalar newFisherWeight);

        double getNegativeLogLikelihoodOfCurrentState();

        std::string getName();

    private:
        StateType computeTruncatedGradient(StateType x);

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
        typename MatrixType::Scalar fisherScale = 1.;
        typename MatrixType::Scalar geometricFactor;
        typename MatrixType::Scalar covarianceFactor;

        std::normal_distribution<typename MatrixType::Scalar> normalDistribution{0., 1.};
        DikinEllipsoidCalculator <MatrixType, VectorType> dikinEllipsoidCalculator;
    };

    template<typename Model,
            typename Matrix>
    CSmMALAProposal<Model, Matrix>::CSmMALAProposal(const Model &model,
                                                    MatrixType A,
                                                    VectorType b,
                                                    VectorType currentState) :
            Model(model),
            A(std::move(A)),
            b(std::move(b)),
            dikinEllipsoidCalculator(this->A, this->b), Model(<#initializer#>, <#initializer#>) {
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
    void CSmMALAProposal<Model, Matrix>::propose(
            RandomNumberGenerator &randomNumberGenerator) {
        for (long i = 0; i < proposal.rows(); ++i) {
            proposal(i) = normalDistribution(randomNumberGenerator);
        }
        proposal = driftedState + covarianceFactor * (stateSqrtInvMetric * proposal);
    }

    template<typename Model, typename Matrix>
    void CSmMALAProposal<Model, Matrix>::acceptProposal() {
        state.swap(proposal);
        driftedState.swap(driftedProposal);
        stateSqrtInvMetric.swap(proposalSqrtInvMetric);
        stateMetric.swap(proposalMetric);
        stateLogSqrtDeterminant = proposalLogSqrtDeterminant;
        stateNegativeLogLikelihood = proposalNegativeLogLikelihood;
    }

    template<typename Model, typename Matrix>
    typename CSmMALAProposal<Model, Matrix>::MatrixType::Scalar
    CSmMALAProposal<Model, Matrix>::computeLogAcceptanceProbability() {
        bool isProposalInteriorPoint = ((A * proposal - b).array() < 0).all();
        if (!isProposalInteriorPoint) {
            return -std::numeric_limits<typename MatrixType::Scalar>::infinity();
        }

        // Important: compute gradient before fisher info or else x3cflux2 will throw
        StateType gradient = computeTruncatedGradient(proposal);
        proposalMetric.setZero();
        if (fisherWeight != 0) {
            auto fisherInformation = fisherScale * Model::computeExpectedFisherInformation(proposal);
            proposalMetric += fisherWeight * fisherInformation;
        }
        if (fisherWeight != 1) {
            auto dikinEllipsoid = dikinEllipsoidCalculator.computeDikinEllipsoid(proposal);
            proposalMetric += (1 - fisherWeight) * dikinEllipsoid;

        }
        CSmMALAProposalDetails::computeMetricInfoForCSmMALAWithSvd(proposalMetric, proposalSqrtInvMetric,
                                                                     proposalLogSqrtDeterminant);
        driftedProposal = proposal +
                          0.5 * std::pow(covarianceFactor, 2) * proposalSqrtInvMetric * proposalSqrtInvMetric *
                          gradient;
        proposalNegativeLogLikelihood = Model::computeNegativeLogLikelihood(proposal);

        double normDifference =
                static_cast<double>((driftedState - proposal).transpose() * stateMetric * (driftedState - proposal)) -
                static_cast<double>((state - driftedProposal).transpose() * proposalMetric * (state - driftedProposal));

        return -proposalNegativeLogLikelihood
               + stateNegativeLogLikelihood
               + proposalLogSqrtDeterminant
               - stateLogSqrtDeterminant
               + geometricFactor * normDifference;
    }

    template<typename Model, typename Matrix>
    typename CSmMALAProposal<Model, Matrix>::StateType
    CSmMALAProposal<Model, Matrix>::getState() const {
        return state;
    }

    template<typename Model, typename Matrix>
    void CSmMALAProposal<Model, Matrix>::setState(StateType newState) {
        state.swap(newState);
        // Important: compute gradient before fisher info or else x3cflux2 will throw
        StateType gradient = computeTruncatedGradient(state);
        stateMetric.setZero();
        if (fisherWeight != 0) {
            auto fisherInformation = fisherScale * Model::computeExpectedFisherInformation(state);
            stateMetric += fisherWeight * fisherInformation;
        }
        if (fisherWeight != 1) {
            auto dikinEllipsoid = dikinEllipsoidCalculator.computeDikinEllipsoid(state);
            stateMetric += (1 - fisherWeight) * dikinEllipsoid;
        }
        CSmMALAProposalDetails::computeMetricInfoForCSmMALAWithSvd(stateMetric,
                                                                     stateSqrtInvMetric,
                                                                     stateLogSqrtDeterminant);
        driftedState = state + 0.5 * std::pow(covarianceFactor, 2) * stateSqrtInvMetric * stateSqrtInvMetric *
                               gradient;
        stateNegativeLogLikelihood = Model::computeNegativeLogLikelihood(state);
    }

    template<typename Model, typename Matrix>
    typename CSmMALAProposal<Model, Matrix>::StateType
    CSmMALAProposal<Model, Matrix>::getProposal() const {
        return proposal;
    }

    template<typename Model, typename Matrix>
    typename CSmMALAProposal<Model, Matrix>::MatrixType::Scalar
    CSmMALAProposal<Model, Matrix>::getStepSize() const {
        return stepSize;
    }

    template<typename Model, typename Matrix>
    void CSmMALAProposal<Model, Matrix>::setStepSize(typename MatrixType::Scalar newStepSize) {
        stepSize = newStepSize;
        geometricFactor = A.cols() / (2 * stepSize * stepSize);
        covarianceFactor = stepSize / std::sqrt(A.cols());
        setState(state);
    }

    template<typename Model, typename Matrix>
    void CSmMALAProposal<Model, Matrix>::setFisherWeight(
            typename MatrixType::Scalar newFisherWeight) {
        if (fisherWeight > 1 || fisherWeight < 0) {
            throw std::runtime_error("fisherWeigth should be in [0, 1].");
        }
        fisherWeight = newFisherWeight;
        setState(state);
    }

    template<typename Model, typename Matrix>
    double CSmMALAProposal<Model, Matrix>::getNegativeLogLikelihoodOfCurrentState() {
        return stateNegativeLogLikelihood;
    }

    template<typename Model, typename Matrix>
    std::string CSmMALAProposal<Model, Matrix>::getName() {
        return "CSmMALA";
    }

    template<typename Model, typename Matrix>
    typename CSmMALAProposal<Model, Matrix>::StateType
    CSmMALAProposal<Model, Matrix>::computeTruncatedGradient(StateType x) {
        StateType gradient = Model::computeLogLikelihoodGradient(x);
        double norm = gradient.norm();
        if (norm != 0) {
            gradient /= norm;
        }
        return gradient;
    }
}

#endif //HOPS_CSMMALA_HPP
