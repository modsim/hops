#ifndef HOPS_CSMMALA_HPP
#define HOPS_CSMMALA_HPP

#include <Eigen/Eigenvalues>
#include <hops/MarkovChain/Proposal/DikinProposal.hpp>
#include <hops/MarkovChain/Recorder/IsAddMessageAvailabe.hpp>
#include <random>
#include <utility>

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

    template<typename ModelType, typename InternalMatrixType>
    class CSmMALAProposal : public ModelType {
    public:
        /**
         * @brief Constructs proposal mechanism on polytope defined as Ax<b.
         * @param A
         * @param b
         * @param currentState
         */
        CSmMALAProposal(ModelType model, InternalMatrixType A, VectorType b, const VectorType& currentState);

        void propose(RandomNumberGenerator &randomNumberGenerator);

        void acceptProposal();

        [[nodiscard]] typename MatrixType::Scalar computeLogAcceptanceProbability();

        VectorType getState() const;

        void setState(VectorType newState);

        VectorType getProposal() const;

        typename MatrixType::Scalar getStepSize() const;

        void setStepSize(typename MatrixType::Scalar newStepSize);

        void setFisherWeight(typename MatrixType::Scalar newFisherWeight);

        double getNegativeLogLikelihoodOfCurrentState();

        std::string getName();

    private:
        VectorType computeTruncatedGradient(VectorType x);

        InternalMatrixType A;
        VectorType b;

        VectorType state;
        VectorType driftedState;
        VectorType proposal;
        VectorType driftedProposal;
        MatrixType::Scalar stateLogSqrtDeterminant = 0;
        MatrixType::Scalar proposalLogSqrtDeterminant = 0;
        MatrixType::Scalar stateNegativeLogLikelihood = 0;
        MatrixType::Scalar proposalNegativeLogLikelihood = 0;
        MatrixType stateSqrtInvMetric;
        MatrixType stateMetric;
        MatrixType proposalSqrtInvMetric;
        MatrixType proposalMetric;

        MatrixType::Scalar stepSize = 1;
        MatrixType::Scalar fisherWeight = .5;
        MatrixType::Scalar fisherScale = 1.;
        MatrixType::Scalar geometricFactor = 0;
        MatrixType::Scalar covarianceFactor = 0;

        std::normal_distribution<MatrixType::Scalar> normalDistribution{0., 1.};
        DikinEllipsoidCalculator <MatrixType, VectorType> dikinEllipsoidCalculator;
    };

    template<typename ModelType, typename InternalMatrixType>
    CSmMALAProposal<ModelType, InternalMatrixType>::CSmMALAProposal(ModelType model,
                                                    InternalMatrixType A,
                                                    hops::VectorType b,
                                                    const VectorType& currentState) :
            ModelType(std::move(model)),
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

    template<typename ModelType, typename InternalMatrixType>
    void CSmMALAProposal<ModelType, InternalMatrixType>::propose(
            RandomNumberGenerator &randomNumberGenerator) {
        for (long i = 0; i < proposal.rows(); ++i) {
            proposal(i) = normalDistribution(randomNumberGenerator);
        }
        proposal = driftedState + covarianceFactor * (stateSqrtInvMetric * proposal);
    }

    template<typename ModelType, typename InternalMatrixType>
    void CSmMALAProposal<ModelType, InternalMatrixType>::acceptProposal() {
        state.swap(proposal);
        driftedState.swap(driftedProposal);
        stateSqrtInvMetric.swap(proposalSqrtInvMetric);
        stateMetric.swap(proposalMetric);
        stateLogSqrtDeterminant = proposalLogSqrtDeterminant;
        stateNegativeLogLikelihood = proposalNegativeLogLikelihood;
    }

    template<typename ModelType, typename InternalMatrixType>
    typename MatrixType::Scalar
    CSmMALAProposal<ModelType, InternalMatrixType>::computeLogAcceptanceProbability() {
        bool isProposalInteriorPoint = ((A * proposal - b).array() < 0).all();
        if (!isProposalInteriorPoint) {
            return -std::numeric_limits<typename MatrixType::Scalar>::infinity();
        }

        // Important: compute gradient before fisher info or else x3cflux2 will throw
        VectorType gradient = computeTruncatedGradient(proposal);
        proposalMetric.setZero();
        if (fisherWeight != 0) {
            auto optionalFisherInformation = ModelType::computeExpectedFisherInformation(proposal);
            if(optionalFisherInformation) {
                auto fisherInformation = optionalFisherInformation.value();
                proposalMetric += (fisherWeight * fisherScale * fisherInformation);
            }
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
        proposalNegativeLogLikelihood = ModelType::computeNegativeLogLikelihood(proposal);

        double normDifference =
                static_cast<double>((driftedState - proposal).transpose() * stateMetric * (driftedState - proposal)) -
                static_cast<double>((state - driftedProposal).transpose() * proposalMetric * (state - driftedProposal));

        return -proposalNegativeLogLikelihood
               + stateNegativeLogLikelihood
               + proposalLogSqrtDeterminant
               - stateLogSqrtDeterminant
               + geometricFactor * normDifference;
    }

    template<typename ModelType, typename InternalMatrixType>
    VectorType CSmMALAProposal<ModelType, InternalMatrixType>::getState() const {
        return state;
    }

    template<typename ModelType, typename InternalMatrixType>
    void CSmMALAProposal<ModelType, InternalMatrixType>::setState(VectorType newState) {
        state.swap(newState);
        // Important: compute gradient before fisher info or else x3cflux2 will throw
        VectorType gradient = computeTruncatedGradient(state);
        stateMetric.setZero();
        if (fisherWeight != 0) {
            auto optionalFisherInformation = ModelType::computeExpectedFisherInformation(proposal);
            if(optionalFisherInformation) {
                auto fisherInformation = optionalFisherInformation.value();
                proposalMetric += fisherWeight * fisherScale * fisherInformation;
            }
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
        stateNegativeLogLikelihood = ModelType::computeNegativeLogLikelihood(state);
    }

    template<typename ModelType, typename InternalMatrixType>
    VectorType CSmMALAProposal<ModelType, InternalMatrixType>::getProposal() const {
        return proposal;
    }

    template<typename ModelType, typename InternalMatrixType>
    MatrixType::Scalar CSmMALAProposal<ModelType, InternalMatrixType>::getStepSize() const {
        return stepSize;
    }

    template<typename ModelType, typename InternalMatrixType>
    void CSmMALAProposal<ModelType, InternalMatrixType>::setStepSize(typename MatrixType::Scalar newStepSize) {
        stepSize = newStepSize;
        geometricFactor = A.cols() / (2 * stepSize * stepSize);
        covarianceFactor = stepSize / std::sqrt(A.cols());
        setState(state);
    }

    template<typename ModelType, typename InternalMatrixType>
    void CSmMALAProposal<ModelType, InternalMatrixType>::setFisherWeight(
            typename MatrixType::Scalar newFisherWeight) {
        if (fisherWeight > 1 || fisherWeight < 0) {
            throw std::runtime_error("fisherWeight should be in [0, 1].");
        }
        fisherWeight = newFisherWeight;
        setState(state);
    }

    template<typename ModelType, typename InternalMatrixType>
    double CSmMALAProposal<ModelType, InternalMatrixType>::getNegativeLogLikelihoodOfCurrentState() {
        return stateNegativeLogLikelihood;
    }

    template<typename ModelType, typename InternalMatrixType>
    std::string CSmMALAProposal<ModelType, InternalMatrixType>::getName() {
        return "CSmMALA";
    }

    template<typename ModelType, typename InternalMatrixType>
    VectorType CSmMALAProposal<ModelType, InternalMatrixType>::computeTruncatedGradient(VectorType x) {
        auto gradient = ModelType::computeLogLikelihoodGradient(x);
        if(gradient) {
            double norm = gradient.value().norm();
            if (norm != 0) {
                gradient.value() /= norm;
            }
            return gradient.value();
        }
        return VectorType::Zero(x.rows());
    }
}

#endif //HOPS_CSMMALA_HPP
