#ifndef HOPS_CSMMALA_HPP
#define HOPS_CSMMALA_HPP

#include <hops/MarkovChain/Proposal/DikinProposal.hpp>
#include <random>

namespace hops {

    template<typename Model, typename Matrix>
    class CSmMALAProposal : public Model {
    private:
    public:
        using MatrixType = Matrix;
        using VectorType = typename Model::VectorType;
        using StateType = typename Model::VectorType;

        /**
         * @brief Constructs Gaussian Dikin proposal mechanism on polytope defined as Ax<b.
         * @param A
         * @param b
         * @param currentState
         */
        CSmMALAProposal(const Model &model, MatrixType A, VectorType b, VectorType currentState);

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
        StateType calculateTruncatedGradient(StateType x);

        MatrixType A;
        VectorType b;

        StateType state;
        StateType driftedState;
        StateType proposal;
        StateType driftedProposal;
        typename MatrixType::Scalar stateLogSqrtDeterminant = 0;
        typename MatrixType::Scalar proposalLogSqrtDeterminant = 0;
        typename MatrixType::Scalar stateLikelihood = 0;
        typename MatrixType::Scalar proposalLikelihood = 0;
        Eigen::Matrix<typename MatrixType::Scalar, Eigen::Dynamic, Eigen::Dynamic> stateCholeskyOfMetric;
        Eigen::Matrix<typename MatrixType::Scalar, Eigen::Dynamic, Eigen::Dynamic> proposalCholeskyOfMetric;

        typename MatrixType::Scalar stepSize;
        typename MatrixType::Scalar fisherWeight = 0.5;
        typename MatrixType::Scalar geometricFactor;
        typename MatrixType::Scalar covarianceFactor;

        std::normal_distribution<typename MatrixType::Scalar> normalDistribution{0., 1.};
        DikinEllipsoidCalculator<MatrixType, VectorType> dikinEllipsoidCalculator;

        Eigen::LLT<Eigen::Matrix<typename MatrixType::Scalar, Eigen::Dynamic, Eigen::Dynamic>> solver;
    };

    template<typename Model, typename Matrix>
    CSmMALAProposal<Model, Matrix>::CSmMALAProposal(const Model &model,
                                                    MatrixType A,
                                                    VectorType b,
                                                    VectorType currentState) :
            Model(model),
            A(std::move(A)),
            b(std::move(b)),
            dikinEllipsoidCalculator(this->A, this->b) {
        setStepSize(1.);
        setState(std::move(currentState));
        proposal = state;
    }

    template<typename Model, typename Matrix>
    void CSmMALAProposal<Model, Matrix>::propose(RandomNumberGenerator &randomNumberGenerator) {
        for (long i = 0; i < proposal.rows(); ++i) {
            proposal(i) = normalDistribution(randomNumberGenerator);
        }
        proposal = driftedState +
                   covarianceFactor * stateCholeskyOfMetric.template triangularView<Eigen::Lower>().solve(proposal);
    }

    template<typename Model, typename Matrix>
    void CSmMALAProposal<Model, Matrix>::acceptProposal() {
        state.swap(proposal);
        if (((A * state - b).array() > 0).any()) {
            throw std::runtime_error("Current state is outside of polytope!");
        }
        driftedState.swap(driftedProposal);
        stateCholeskyOfMetric.swap(proposalCholeskyOfMetric);
        stateLogSqrtDeterminant = proposalLogSqrtDeterminant;
        stateLikelihood = proposalLikelihood;
    }

    template<typename Model, typename Matrix>
    typename CSmMALAProposal<Model, Matrix>::MatrixType::Scalar
    CSmMALAProposal<Model, Matrix>::calculateLogAcceptanceProbability() {
        bool isProposalInteriorPoint = ((A * proposal - b).array() <= 0).all();
        if (!isProposalInteriorPoint) {
            return -std::numeric_limits<typename MatrixType::Scalar>::infinity();
        }

        proposalLikelihood = Model::calculateNegativeLogLikelihood(proposal);
        proposalCholeskyOfMetric = solver.compute(
                        fisherWeight * Model::calculateExpectedFisherInformation(proposal)
                        + (1 - fisherWeight) * dikinEllipsoidCalculator.calculateDikinEllipsoid(proposal))
                .matrixL();
        proposalLogSqrtDeterminant = proposalCholeskyOfMetric.diagonal().array().log().sum();
        driftedProposal = proposal + 0.5 * std::pow(covarianceFactor, 2) *
                                     proposalCholeskyOfMetric.template triangularView<Eigen::Lower>().solve(
                                             proposalCholeskyOfMetric.transpose().template triangularView<Eigen::Lower>().solve(
                                                     calculateTruncatedGradient(proposal)
                                             )
                                     );

        return -proposalLikelihood
               + stateLikelihood
               + proposalLogSqrtDeterminant
               - stateLogSqrtDeterminant
               + geometricFactor * ((stateCholeskyOfMetric * (driftedState - proposal)).squaredNorm()
                                    - (proposalCholeskyOfMetric * (state - driftedProposal)).squaredNorm()
        );
    }

    template<typename Model, typename Matrix>
    typename CSmMALAProposal<Model, Matrix>::StateType
    CSmMALAProposal<Model, Matrix>::getState() const {
        return state;
    }

    template<typename Model, typename Matrix>
    void CSmMALAProposal<Model, Matrix>::setState(StateType newState) {
        state.swap(newState);
        stateCholeskyOfMetric = solver.compute(
                        fisherWeight * Model::calculateExpectedFisherInformation(state) +
                        (1 - fisherWeight) * dikinEllipsoidCalculator.calculateDikinEllipsoid(state))
                .matrixL();
        stateLogSqrtDeterminant = stateCholeskyOfMetric.diagonal().array().log().sum();
        stateLikelihood = Model::calculateNegativeLogLikelihood(state);

        driftedState = state + 0.5 * std::pow(covarianceFactor, 2) *
                               stateCholeskyOfMetric.template triangularView<Eigen::Lower>().solve(
                                       stateCholeskyOfMetric.transpose().template triangularView<Eigen::Lower>().solve(
                                               calculateTruncatedGradient(state)
                                       )
                               );
    }

    template<typename Model, typename Matrix>
    typename CSmMALAProposal<Model, Matrix>::StateType
    CSmMALAProposal<Model, Matrix>::getProposal() const {
        return proposal;
    }

    template<typename Model, typename Matrix>
    typename CSmMALAProposal<Model, Matrix>::MatrixType::Scalar CSmMALAProposal<Model, Matrix>::getStepSize() const {
        return stepSize;
    }

    template<typename Model, typename Matrix>
    void CSmMALAProposal<Model, Matrix>::setStepSize(typename MatrixType::Scalar newStepSize) {
        stepSize = newStepSize;
        geometricFactor = A.cols() / (2 * stepSize);
        covarianceFactor = std::sqrt(stepSize / A.cols());
    }

    template<typename Model, typename Matrix>
    typename CSmMALAProposal<Model, Matrix>::StateType
    CSmMALAProposal<Model, Matrix>::calculateTruncatedGradient(StateType x) {
        StateType gradient = Model::calculateLogLikelihoodGradient(x);
        double norm = gradient.norm();
        if (norm > 1) {
            gradient /= norm;
        }
        return gradient;
    }

    template<typename Model, typename Matrix>
    double CSmMALAProposal<Model, Matrix>::getNegativeLogLikelihoodOfCurrentState() {
        return stateLikelihood;
    }

    template<typename Model, typename Matrix>
    std::string CSmMALAProposal<Model, Matrix>::getName() {
        return "CSmMALA";
    }
}

#endif //HOPS_CSMMALA_HPP
