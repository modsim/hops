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
    class CSmMALAProposal : public Proposal, public ModelType {
    public:
        /**
         * @brief Constructs proposal mechanism on polytope defined as Ax<b.
         * @param A
         * @param b
         * @param currentState
         * @param fisherWeight parameterizes the mixing of Dikin metric and Fisher information.
         */
        CSmMALAProposal(ModelType model,
                        InternalMatrixType A,
                        VectorType b,
                        const VectorType &currentState,
                        double newFisherWeight = 0.5);

        std::pair<double, VectorType> propose(RandomNumberGenerator &rng) override;

        VectorType acceptProposal() override;

        void setState(VectorType state) override;

        [[nodiscard]] VectorType getState() const override;

        [[nodiscard]] VectorType getProposal() const override;

        void setParameter(ProposalParameterName parameterName, const std::any &value) override;

        [[nodiscard]] std::optional<double> getStepSize() const;

        void setStepSize(double stepSize);

        [[nodiscard]] bool hasStepSize() const override;

        [[nodiscard]] std::string getProposalName() const override;

        [[nodiscard]] double getStateNegativeLogLikelihood() const override;

        [[nodiscard]] std::unique_ptr<Proposal> deepCopy() const override;

        [[nodiscard]] double computeLogAcceptanceProbability();

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
                                                                    const VectorType &currentState,
                                                                    double newFisherWeight) :
            ModelType(std::move(model)),
            A(std::move(A)),
            b(std::move(b)),
            dikinEllipsoidCalculator(this->A, this->b) {
        if (newFisherWeight > 1 || newFisherWeight < 0) {
            throw std::invalid_argument("fisherWeight should be in [0, 1].");
        }
        this->fisherWeight = newFisherWeight;

        stateMetric = Eigen::Matrix<typename MatrixType::Scalar, Eigen::Dynamic, Eigen::Dynamic>::Zero(
                currentState.rows(), currentState.rows());
        proposalMetric = Eigen::Matrix<typename MatrixType::Scalar, Eigen::Dynamic, Eigen::Dynamic>::Zero(
                currentState.rows(), currentState.rows());
        CSmMALAProposal::setState(currentState);
        CSmMALAProposal::setStepSize(1.);
        proposal = state;
    }

    template<typename ModelType, typename InternalMatrixType>
    std::pair<double, VectorType> CSmMALAProposal<ModelType, InternalMatrixType>::propose(RandomNumberGenerator &rng) {
        for (long i = 0; i < proposal.rows(); ++i) {
            proposal(i) = normalDistribution(rng);
        }
        proposal = driftedState + covarianceFactor * (stateSqrtInvMetric * proposal);

       return {computeLogAcceptanceProbability(), proposal};
    }

    template<typename ModelType, typename InternalMatrixType>
    VectorType CSmMALAProposal<ModelType, InternalMatrixType>::acceptProposal() {
        state.swap(proposal);
        driftedState.swap(driftedProposal);
        stateSqrtInvMetric.swap(proposalSqrtInvMetric);
        stateMetric.swap(proposalMetric);
        stateLogSqrtDeterminant = proposalLogSqrtDeterminant;
        stateNegativeLogLikelihood = proposalNegativeLogLikelihood;
        return state;
    }

    template<typename ModelType, typename InternalMatrixType>
    void CSmMALAProposal<ModelType, InternalMatrixType>::setState(VectorType newState) {
        state.swap(newState);
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
            std::optional<decltype(proposalMetric)> optionalFisherInformation = ModelType::computeExpectedFisherInformation(proposal);
            if(optionalFisherInformation) {
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

        return // TODO remove likelihoods here -proposalNegativeLogLikelihood + stateNegativeLogLikelihood
               + proposalLogSqrtDeterminant
               - stateLogSqrtDeterminant
               + geometricFactor * normDifference;
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

    template<typename ModelType, typename InternalMatrixType>
    bool CSmMALAProposal<ModelType, InternalMatrixType>::hasStepSize() const {
        return true;
    }

    template<typename ModelType, typename InternalMatrixType>
    std::unique_ptr<Proposal> CSmMALAProposal<ModelType, InternalMatrixType>::deepCopy() const {
        // TODO check if we need to clone model, probably we do!
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
    void CSmMALAProposal<ModelType, InternalMatrixType>::setParameter(ProposalParameterName parameterName,
                                                                      const std::any &value) {
        switch (parameterName) {
            case ProposalParameterName::STEP_SIZE: {
                setStepSize(std::any_cast<double>(value));
                break;
            }
            default:
                throw std::invalid_argument("Can't set parameter which doesn't exist in CSmMALAProposal.");
        }

    }
}

#endif //HOPS_CSMMALA_HPP
