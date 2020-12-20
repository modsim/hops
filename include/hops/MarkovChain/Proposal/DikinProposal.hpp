#ifndef HOPS_DIKINPROPOSAL_HPP
#define HOPS_DIKINPROPOSAL_HPP

#include <Eigen/LU>
#include <hops/MarkovChain/Proposal/DikinEllipsoidCalculator.hpp>
#include <hops/RandomNumberGenerator/RandomNumberGenerator.hpp>
#include <random>

namespace hops {
    namespace DikinProposalDetails {
        template<typename MatrixType1, typename MatrixType2, typename VectorType>
        void calculateDikinValuesWithSvd(const VectorType &state,
                                         const hops::DikinEllipsoidCalculator<MatrixType1, VectorType> &dikinEllipsoidCalculator,
                                         MatrixType2 &dikinEllipsoid,
                                         MatrixType2 &sqrtInvDikinEllipsoid,
                                         double &logSqrtDeterminant) {
            dikinEllipsoid = dikinEllipsoidCalculator.calculateDikinEllipsoid(state);
            Eigen::BDCSVD<MatrixType2> solver(MatrixType2(dikinEllipsoid), Eigen::ComputeFullU | Eigen::ComputeFullV);
            sqrtInvDikinEllipsoid = solver.matrixU() * solver.singularValues().cwiseInverse().cwiseSqrt().asDiagonal() *
                                    solver.matrixU().adjoint();
            logSqrtDeterminant = 0.5 * solver.singularValues().array().log().sum();
        }
    }

    template<typename MatrixType, typename VectorType>
    class DikinProposal {
    public:
        using StateType = VectorType;

        /**
         * @brief Constructs Gaussian Dikin proposal mechanism on polytope defined as Ax<b.
         * @param A
         * @param b
         * @param currentState
         */
        DikinProposal(MatrixType A, VectorType b, VectorType currentState);

        void propose(RandomNumberGenerator &randomNumberGenerator);

        void acceptProposal();

        typename MatrixType::Scalar calculateLogAcceptanceProbability();

        StateType getState() const;

        void setState(StateType newState);

        StateType getProposal() const;

        typename MatrixType::Scalar getStepSize() const;

        void setStepSize(typename MatrixType::Scalar newStepSize);

        std::string getName();

    private:
        MatrixType A;
        VectorType b;

        StateType state;
        StateType proposal;
        typename MatrixType::Scalar stateLogSqrtDeterminant = 0;
        typename MatrixType::Scalar proposalLogSqrtDeterminant = 0;
        Eigen::Matrix<typename MatrixType::Scalar, Eigen::Dynamic, Eigen::Dynamic> stateSqrtInvDikinEllipsoid;
        Eigen::Matrix<typename MatrixType::Scalar, Eigen::Dynamic, Eigen::Dynamic> stateDikinEllipsoid;
        Eigen::Matrix<typename MatrixType::Scalar, Eigen::Dynamic, Eigen::Dynamic> proposalSqrtInvDikinEllipsoid;
        Eigen::Matrix<typename MatrixType::Scalar, Eigen::Dynamic, Eigen::Dynamic> proposalDikinEllipsoid;

        typename MatrixType::Scalar stepSize;
        typename MatrixType::Scalar geometricFactor;
        typename MatrixType::Scalar covarianceFactor;

        std::normal_distribution<typename MatrixType::Scalar> normalDistribution{0., 1.};
        DikinEllipsoidCalculator<MatrixType, VectorType> dikinEllipsoidCalculator;
    };

    template<typename MatrixType, typename VectorType>
    DikinProposal<MatrixType, VectorType>::DikinProposal(MatrixType A,
                                                         VectorType b,
                                                         VectorType currentState) :
            A(std::move(A)),
            b(std::move(b)),
            dikinEllipsoidCalculator(this->A, this->b) {
        setState(currentState);
        setStepSize(1);
        proposal = state;
    }

    template<typename MatrixType, typename VectorType>
    void DikinProposal<MatrixType, VectorType>::propose(
            RandomNumberGenerator &randomNumberGenerator) {
        for (long i = 0; i < proposal.rows(); ++i) {
            proposal(i) = normalDistribution(randomNumberGenerator);
        }
        proposal = state + covarianceFactor * (stateSqrtInvDikinEllipsoid * proposal);
    }

    template<typename MatrixType, typename VectorType>
    void DikinProposal<MatrixType, VectorType>::acceptProposal() {
        state.swap(proposal);
        stateDikinEllipsoid.swap(proposalDikinEllipsoid);
        stateSqrtInvDikinEllipsoid.swap(proposalSqrtInvDikinEllipsoid);
        stateLogSqrtDeterminant = proposalLogSqrtDeterminant;
    }

    template<typename MatrixType, typename VectorType>
    typename MatrixType::Scalar
    DikinProposal<MatrixType, VectorType>::calculateLogAcceptanceProbability() {
        bool isProposalInteriorPoint = ((A * proposal - b).array() < 0).all();
        if (!isProposalInteriorPoint) {
            return -std::numeric_limits<typename MatrixType::Scalar>::infinity();
        }

        DikinProposalDetails::calculateDikinValuesWithSvd(proposal,
                                dikinEllipsoidCalculator,
                                proposalDikinEllipsoid,
                                proposalSqrtInvDikinEllipsoid,
                                proposalLogSqrtDeterminant);

        VectorType stateDifference = state - proposal;
        double normDifference =
                stateDifference.transpose() * (stateDikinEllipsoid - proposalDikinEllipsoid) * stateDifference;

        return proposalLogSqrtDeterminant
               - stateLogSqrtDeterminant
               + geometricFactor * normDifference;
    }

    template<typename MatrixType, typename VectorType>
    typename DikinProposal<MatrixType, VectorType>::StateType
    DikinProposal<MatrixType, VectorType>::getState() const {
        return state;
    }

    template<typename MatrixType, typename VectorType>
    void DikinProposal<MatrixType, VectorType>::setState(StateType newState) {
        state.swap(newState);
        DikinProposalDetails::calculateDikinValuesWithSvd(state,
                                dikinEllipsoidCalculator,
                                stateDikinEllipsoid,
                                stateSqrtInvDikinEllipsoid,
                                stateLogSqrtDeterminant);
    }

    template<typename MatrixType, typename VectorType>
    typename DikinProposal<MatrixType, VectorType>::StateType
    DikinProposal<MatrixType, VectorType>::getProposal() const {
        return proposal;
    }

    template<typename MatrixType, typename VectorType>
    typename MatrixType::Scalar DikinProposal<MatrixType, VectorType>::getStepSize() const {
        return stepSize;
    }

    template<typename MatrixType, typename VectorType>
    void DikinProposal<MatrixType, VectorType>::setStepSize(
            typename MatrixType::Scalar newStepSize) {
        stepSize = newStepSize;
        geometricFactor = A.cols() / (2 * stepSize * stepSize);
        covarianceFactor = stepSize / std::sqrt(A.cols());
        setState(state);
    }

    template<typename MatrixType, typename VectorType>
    std::string DikinProposal<MatrixType, VectorType>::getName() {
        return "Dikin Walk";
    }
}

#endif //HOPS_DIKINPROPOSAL_HPP
