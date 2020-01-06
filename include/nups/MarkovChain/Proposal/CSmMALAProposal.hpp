#ifndef NUPS_CSMMALA_HPP
#define NUPS_CSMMALA_HPP

#include <nups/MarkovChain/Proposal/DikinProposal.hpp>
#include <random>

namespace nups {
    template<typename Model>
    class CSmMALAProposal {
    private:
    public:
        using MatrixType = typename Model::MatrixType;
        using VectorType = typename Model::VectorType;
        using StateType = typename Model::VectorType;

        /**
         * @brief Constructs Gaussian Dikin proposal mechanism on polytope defined as Ax<b.
         * @param A
         * @param b
         * @param currentState
         */
        CSmMALAProposal(const Model &model, MatrixType A, VectorType b, StateType currentState);

        void propose(RandomNumberGenerator &randomNumberGenerator);

        void acceptProposal();

        [[nodiscard]] typename MatrixType::Scalar calculateLogAcceptanceProbability();

        StateType getState() const;

        void setState(StateType newState);

        StateType getProposal() const;

        typename MatrixType::Scalar getStepSize() const;

        void setStepSize(typename MatrixType::Scalar newStepSize);

    private:
        Model model;
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
        DikinEllipsoidCalculator <MatrixType, VectorType> dikinEllipsoidCalculator;

        Eigen::LLT<Eigen::Matrix<typename MatrixType::Scalar, Eigen::Dynamic, Eigen::Dynamic>> solver;
    };

    template<typename Model>
    CSmMALAProposal<Model>::CSmMALAProposal(const Model &model,
                                            MatrixType A,
                                            VectorType b,
                                            StateType currentState) :
            model(model),
            A(std::move(A)),
            b(std::move(b)),
            dikinEllipsoidCalculator(this->A, this->b) {
        setStepSize(1.);
        setState(std::move(currentState));
        proposal = state;
    }

    template<typename Model>
    void CSmMALAProposal<Model>::propose(RandomNumberGenerator &randomNumberGenerator) {
        for (long i = 0; i < proposal.rows(); ++i) {
            proposal(i) = normalDistribution(randomNumberGenerator);
        }
        proposal = driftedState +
                   covarianceFactor * stateCholeskyOfMetric.template triangularView<Eigen::Upper>().solve(proposal);
    }

    template<typename Model>
    void CSmMALAProposal<Model>::acceptProposal() {
        state.swap(proposal);
        driftedState.swap(driftedProposal);
        stateCholeskyOfMetric.swap(proposalCholeskyOfMetric);
        stateLogSqrtDeterminant = proposalLogSqrtDeterminant;
        stateLikelihood = proposalLikelihood;
    }

    template<typename Model>
    typename CSmMALAProposal<Model>::MatrixType::Scalar CSmMALAProposal<Model>::calculateLogAcceptanceProbability() {
        bool isProposalInteriorPoint = ((A * proposal - b).array() <= 0).all();
        if (!isProposalInteriorPoint) {
            return -std::numeric_limits<typename MatrixType::Scalar>::infinity();
        }

        proposalLikelihood = model.calculateNegativeLogLikelihood(proposal);
        proposalCholeskyOfMetric = solver.compute(
                        fisherWeight * model.calculateExpectedFisherInformation(proposal)
                        + (1 - fisherWeight) * dikinEllipsoidCalculator.calculateDikinEllipsoid(proposal))
                .matrixU();
        proposalLogSqrtDeterminant = proposalCholeskyOfMetric.diagonal().array().log().sum();
        driftedProposal = proposal + 0.5 * std::pow(covarianceFactor, 2) *
                                     proposalCholeskyOfMetric.template triangularView<Eigen::Upper>().solve(
                                             proposalCholeskyOfMetric.transpose().template triangularView<Eigen::Upper>().solve(
                                                     model.calculateGradient(proposal)
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

    template<typename Model>
    typename CSmMALAProposal<Model>::StateType
    CSmMALAProposal<Model>::getState() const {
        return state;
    }

    template<typename Model>
    void CSmMALAProposal<Model>::setState(StateType newState) {
        state.swap(newState);
        stateCholeskyOfMetric = solver.compute(
                        fisherWeight * model.calculateExpectedFisherInformation(state) +
                        (1 - fisherWeight) * dikinEllipsoidCalculator.calculateDikinEllipsoid(state))
                .matrixU();
        stateLogSqrtDeterminant = stateCholeskyOfMetric.diagonal().array().log().sum();
        stateLikelihood = model.calculateNegativeLogLikelihood(state);

        driftedState = state + 0.5 * std::pow(covarianceFactor, 2) *
                               stateCholeskyOfMetric.template triangularView<Eigen::Upper>().solve(
                                       stateCholeskyOfMetric.transpose().template triangularView<Eigen::Upper>().solve(
                                               model.calculateGradient(state)
                                       )
                               );
    }

    template<typename Model>
    typename CSmMALAProposal<Model>::StateType
    CSmMALAProposal<Model>::getProposal() const {
        return proposal;
    }

    template<typename Model>
    typename CSmMALAProposal<Model>::MatrixType::Scalar CSmMALAProposal<Model>::getStepSize() const {
        return stepSize;
    }

    template<typename Model>
    void CSmMALAProposal<Model>::setStepSize(typename MatrixType::Scalar newStepSize) {
        stepSize = newStepSize;
        geometricFactor = A.cols() / (2 * stepSize);
        covarianceFactor = std::sqrt(stepSize / A.cols());
    }
}

#endif //NUPS_CSMMALA_HPP
