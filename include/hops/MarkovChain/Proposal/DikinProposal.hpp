#ifndef HOPS_DIKINPROPOSAL_HPP
#define HOPS_DIKINPROPOSAL_HPP

#include <Eigen/LU>
#include <hops/MarkovChain/Proposal/DikinEllipsoidCalculator.hpp>
#include <hops/RandomNumberGenerator/RandomNumberGenerator.hpp>
#include <random>

namespace hops {
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

        [[nodiscard]] typename MatrixType::Scalar calculateLogAcceptanceProbability();

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
        Eigen::Matrix<typename MatrixType::Scalar, Eigen::Dynamic, Eigen::Dynamic> stateCholeskyOfDikinEllipsoid;
        Eigen::Matrix<typename MatrixType::Scalar, Eigen::Dynamic, Eigen::Dynamic> proposalCholeskyOfDikinEllipsoid;

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
        setStepSize(1.);
        setState(std::move(currentState));
        proposal = state;
    }

    template<typename MatrixType, typename VectorType>
    void DikinProposal<MatrixType, VectorType>::propose(RandomNumberGenerator &randomNumberGenerator) {
        for (long i = 0; i < proposal.rows(); ++i) {
            proposal(i) = normalDistribution(randomNumberGenerator);
        }
        proposal = state + covarianceFactor *
                           stateCholeskyOfDikinEllipsoid.template triangularView<Eigen::Lower>().solve(proposal);
    }

    template<typename MatrixType, typename VectorType>
    void DikinProposal<MatrixType, VectorType>::acceptProposal() {
        state.swap(proposal);
        stateCholeskyOfDikinEllipsoid = std::move(proposalCholeskyOfDikinEllipsoid);
        stateLogSqrtDeterminant = proposalLogSqrtDeterminant;
    }

    template<typename MatrixType, typename VectorType>
    typename MatrixType::Scalar
    DikinProposal<MatrixType, VectorType>::calculateLogAcceptanceProbability() {
        bool isProposalInteriorPoint = ((A * proposal - b).array() <= 0).all();
        if (!isProposalInteriorPoint) {
            return -std::numeric_limits<typename MatrixType::Scalar>::infinity();
        }

        proposalCholeskyOfDikinEllipsoid = dikinEllipsoidCalculator.calculateCholeskyFactorOfDikinEllipsoid(proposal);
        proposalLogSqrtDeterminant = proposalCholeskyOfDikinEllipsoid.diagonal().array().log().sum();
        VectorType stateDifference = state - proposal;

        return proposalLogSqrtDeterminant
               - stateLogSqrtDeterminant
               + geometricFactor * ((stateCholeskyOfDikinEllipsoid * stateDifference).squaredNorm()
                                    - (proposalCholeskyOfDikinEllipsoid * stateDifference).squaredNorm()
        );
    }

    template<typename MatrixType, typename VectorType>
    typename DikinProposal<MatrixType, VectorType>::StateType
    DikinProposal<MatrixType, VectorType>::getState() const {
        return state;
    }

    template<typename MatrixType, typename VectorType>
    void DikinProposal<MatrixType, VectorType>::setState(StateType newState) {
        state.swap(newState);
        stateCholeskyOfDikinEllipsoid = dikinEllipsoidCalculator.calculateCholeskyFactorOfDikinEllipsoid(state);
        stateLogSqrtDeterminant = stateCholeskyOfDikinEllipsoid.diagonal().array().log().sum();
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
    void DikinProposal<MatrixType, VectorType>::setStepSize(typename MatrixType::Scalar newStepSize) {
        stepSize = newStepSize;
        geometricFactor = A.cols() / (2 * stepSize);
        covarianceFactor = std::sqrt(stepSize / A.cols());
    }

    template<typename MatrixType, typename VectorType>
    std::string DikinProposal<MatrixType, VectorType>::getName() {
        return "Dikin Walk";
    }
}

#endif //HOPS_DIKINPROPOSAL_HPP
