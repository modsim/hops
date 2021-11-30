#ifndef HOPS_DIKINPROPOSAL_HPP
#define HOPS_DIKINPROPOSAL_HPP

#include <Eigen/LU>
#include "DikinEllipsoidCalculator.hpp"
#include "../../RandomNumberGenerator/RandomNumberGenerator.hpp"
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

        typename MatrixType::Scalar computeLogAcceptanceProbability();

        StateType getState() const;

        void setState(StateType newState);

        StateType getProposal() const;

        typename MatrixType::Scalar getStepSize() const;

        void setStepSize(typename MatrixType::Scalar newStepSize);

        std::string getName();


    protected:
        StateType state;
        StateType proposal;

    private:
        MatrixType A;
        VectorType b;

        typename MatrixType::Scalar stateLogSqrtDeterminant = 0;
        typename MatrixType::Scalar proposalLogSqrtDeterminant = 0;
        Eigen::Matrix<typename MatrixType::Scalar, Eigen::Dynamic, Eigen::Dynamic> stateCholeskyOfDikinEllipsoid;
        Eigen::Matrix<typename MatrixType::Scalar, Eigen::Dynamic, Eigen::Dynamic> proposalCholeskyOfDikinEllipsoid;

        typename MatrixType::Scalar stepSize = 0.075; // value  from dikin walk publication
        typename MatrixType::Scalar geometricFactor;
        typename MatrixType::Scalar covarianceFactor;
        constexpr static typename MatrixType::Scalar boundaryCushion = 0;

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
    DikinProposal<MatrixType, VectorType>::computeLogAcceptanceProbability() {
        bool isProposalInteriorPoint = ((A * proposal - b).array() < -boundaryCushion).all();
        if (!isProposalInteriorPoint) {
            return -std::numeric_limits<typename MatrixType::Scalar>::infinity();
        }

        auto choleskyResult = dikinEllipsoidCalculator.computeCholeskyFactorOfDikinEllipsoid(proposal);
        if (!choleskyResult.first) {
            return -std::numeric_limits<typename MatrixType::Scalar>::infinity();
        }
        proposalCholeskyOfDikinEllipsoid = std::move(choleskyResult.second);

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
        auto choleskyResult = dikinEllipsoidCalculator.computeCholeskyFactorOfDikinEllipsoid(state);
        if (!choleskyResult.first) {
            throw std::runtime_error("Could not compute cholesky factorization for newState.");
        }
        stateCholeskyOfDikinEllipsoid = std::move(choleskyResult.second);
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
