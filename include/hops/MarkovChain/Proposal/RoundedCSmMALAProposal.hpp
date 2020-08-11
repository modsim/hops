#ifndef HOPS_ROUNDEDCSMMALA_HPP
#define HOPS_ROUNDEDCSMMALA_HPP

#include <hops/MarkovChain/Proposal/DikinProposal.hpp>
#include <hops/MarkovChain/Recorder/IsStoreMetropolisHastingsInfoRecordAvailable.hpp>
#include <random>

namespace hops {

    template<typename Model, typename Matrix>
    class RoundedCSmMALAProposal : public Model {
    private:
    public:
        using MatrixType = Matrix;
        using VectorType = typename Model::VectorType;
        using StateType = typename Model::VectorType;

        /**
         * @brief Constructs proposal mechanism on polytope defined as roundedA*x<roundedB.
         * @param A
         * @param roundedB
         * @param currentState
         * @param rounding transformation (e.g. cholesky factor L of maximum volume ellipsoid)
         */
        RoundedCSmMALAProposal(const Model &model, MatrixType A, VectorType roundedB, VectorType currentState,
                               MatrixType rounding);

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
        MatrixType rounding;

        StateType state;
        StateType driftedState;
        StateType proposal;
        StateType driftedProposal;
        StateType unroundedState;
        StateType unroundedProposal;

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
        constexpr static typename MatrixType::Scalar boundaryCushion = 1e-10;
    };

    template<typename Model, typename Matrix>
    RoundedCSmMALAProposal<Model, Matrix>::RoundedCSmMALAProposal(const Model &model,
                                                                  MatrixType A,
                                                                  VectorType b,
                                                                  VectorType currentState,
                                                                  MatrixType rounding) :
            Model(model),
            A(std::move(A)),
            b(std::move(b)),
            rounding(std::move(rounding)),
            dikinEllipsoidCalculator(this->A * this->rounding, this->b) {
        setStepSize(1.);
        setState(std::move(currentState));
        proposal = state;
        unroundedProposal = unroundedState;
    }

    template<typename Model, typename Matrix>
    void RoundedCSmMALAProposal<Model, Matrix>::propose(RandomNumberGenerator &randomNumberGenerator) {
        for (long i = 0; i < proposal.rows(); ++i) {
            proposal(i) = normalDistribution(randomNumberGenerator);
        }
        proposal = driftedState +
                   covarianceFactor * stateCholeskyOfMetric.template triangularView<Eigen::Lower>().solve(proposal);
        unroundedProposal = rounding * proposal;
    }

    template<typename Model, typename Matrix>
    void RoundedCSmMALAProposal<Model, Matrix>::acceptProposal() {
        state.swap(proposal);
        unroundedState.swap(unroundedProposal);
        if (((A * unroundedState - b).array() > boundaryCushion).any()) {
            throw std::runtime_error("Current state is outside of polytope!");
        }
        driftedState.swap(driftedProposal);
        stateCholeskyOfMetric.swap(proposalCholeskyOfMetric);
        stateLogSqrtDeterminant = proposalLogSqrtDeterminant;
        stateLikelihood = proposalLikelihood;
    }

    template<typename Model, typename Matrix>
    typename RoundedCSmMALAProposal<Model, Matrix>::MatrixType::Scalar
    RoundedCSmMALAProposal<Model, Matrix>::calculateLogAcceptanceProbability() {
        bool isProposalInteriorPoint = ((A * unroundedProposal - b).array() < -boundaryCushion).all();
        if (!isProposalInteriorPoint) {
            if constexpr(IsStoreMetropolisHastingsInfoRecordAvailable<Model>::value) {
                Model::storeMetropolisHastingsInfoRecord("polytope");
            }
            return -std::numeric_limits<typename MatrixType::Scalar>::infinity();
        }
        if constexpr(IsStoreMetropolisHastingsInfoRecordAvailable<Model>::value) {
            Model::storeMetropolisHastingsInfoRecord("likelihood");
        }

        proposalLikelihood = Model::calculateNegativeLogLikelihood(unroundedProposal);
        proposalCholeskyOfMetric = solver.compute(
                        fisherWeight * (
                                rounding.transpose() *
                                Model::calculateExpectedFisherInformation(unroundedProposal) *
                                rounding
                        )
                        + (1 - fisherWeight) * dikinEllipsoidCalculator.calculateDikinEllipsoid(proposal)
                )
                .matrixL();
        proposalCholeskyOfMetric = solver.matrixL();

        proposalLogSqrtDeterminant = proposalCholeskyOfMetric.diagonal().array().log().sum();
        driftedProposal = proposal + 0.5 * std::pow(covarianceFactor, 2) *
                                     proposalCholeskyOfMetric.transpose().template triangularView<Eigen::Upper>().solve(
                                             proposalCholeskyOfMetric.template triangularView<Eigen::Lower>().solve(
                                                     rounding.transpose() *
                                                     calculateTruncatedGradient(unroundedProposal)
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
    typename RoundedCSmMALAProposal<Model, Matrix>::StateType
    RoundedCSmMALAProposal<Model, Matrix>::getState() const {
        return rounding * state;
    }

    template<typename Model, typename Matrix>
    void RoundedCSmMALAProposal<Model, Matrix>::setState(StateType newState) {
        unroundedState = newState;
        state = rounding.template triangularView<Eigen::Lower>().solve(unroundedState);

        stateCholeskyOfMetric = solver.compute(
                        fisherWeight * (
                                rounding.transpose() *
                                Model::calculateExpectedFisherInformation(unroundedState) *
                                rounding
                        ) +
                        (1 - fisherWeight) * dikinEllipsoidCalculator.calculateDikinEllipsoid(state)
                )
                .matrixL();
        stateLogSqrtDeterminant = stateCholeskyOfMetric.diagonal().array().log().sum();
        stateLikelihood = Model::calculateNegativeLogLikelihood(unroundedState);


        driftedState = state + 0.5 * std::pow(covarianceFactor, 2) *
                               stateCholeskyOfMetric.transpose().template triangularView<Eigen::Upper>().solve(
                                       stateCholeskyOfMetric.template triangularView<Eigen::Lower>().solve(
                                               rounding.transpose() * calculateTruncatedGradient(unroundedState)
                                       )
                               );
    }

    template<typename Model, typename Matrix>
    typename RoundedCSmMALAProposal<Model, Matrix>::StateType
    RoundedCSmMALAProposal<Model, Matrix>::getProposal() const {
        return rounding * proposal;
    }

    template<typename Model, typename Matrix>
    typename RoundedCSmMALAProposal<Model, Matrix>::MatrixType::Scalar
    RoundedCSmMALAProposal<Model, Matrix>::getStepSize() const {
        return stepSize;
    }

    template<typename Model, typename Matrix>
    void RoundedCSmMALAProposal<Model, Matrix>::setStepSize(typename MatrixType::Scalar newStepSize) {
        stepSize = newStepSize;
        geometricFactor = A.cols() / (2 * stepSize);
        covarianceFactor = std::sqrt(stepSize / A.cols());
    }

    template<typename Model, typename Matrix>
    typename RoundedCSmMALAProposal<Model, Matrix>::StateType
    RoundedCSmMALAProposal<Model, Matrix>::calculateTruncatedGradient(StateType x) {
        StateType gradient = Model::calculateLogLikelihoodGradient(x);
        double norm = gradient.norm();
        if (norm > 1) {
            gradient /= norm;
        }
        return gradient;
    }

    template<typename Model, typename Matrix>
    double RoundedCSmMALAProposal<Model, Matrix>::getNegativeLogLikelihoodOfCurrentState() {
        return stateLikelihood;
    }

    template<typename Model, typename Matrix>
    std::string RoundedCSmMALAProposal<Model, Matrix>::getName() {
        return "Rounded CSmMALA";
    }
}

#endif //HOPS_ROUNDEDCSMMALA_HPP
