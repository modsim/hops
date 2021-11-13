#ifndef HOPS_DIKINPROPOSAL_HPP
#define HOPS_DIKINPROPOSAL_HPP

#include <Eigen/LU>
#include <random>

#include <hops/RandomNumberGenerator/RandomNumberGenerator.hpp>

#include "Proposal.hpp"
#include "DikinEllipsoidCalculator.hpp"

namespace hops {
    template<typename InternalMatrixType, typename InternalVectorType>
    class DikinProposal : public Proposal {
    public:
        /**
         * @brief Constructs Gaussian Dikin proposal mechanism on polytope defined as Ax<b.
         * @param A
         * @param b
         * @param currentState
         * @param stepSize radius of dikin ellipsoids. Default is from https://doi.org/10.1287/moor.1110.0519.
         */
        DikinProposal(InternalMatrixType A, InternalVectorType b, const VectorType& currentState, double stepSize = 0.075);

        std::pair<double, VectorType> propose(RandomNumberGenerator &randomNumberGenerator) override;

        VectorType acceptProposal() override;

        void setState(VectorType newState) override;

        void setStepSize(double stepSize) override;

        [[nodiscard]] std::string getProposalName() const override;

        [[nodiscard]] std::optional<double> getStepSize() const override;

        bool hasStepSize() const override;

        std::unique_ptr<Proposal> deepCopy() const override;


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

        typename MatrixType::Scalar stepSize = 0.075; // value from dikin walk publication
        typename MatrixType::Scalar geometricFactor;
        typename MatrixType::Scalar covarianceFactor;
        constexpr static typename MatrixType::Scalar boundaryCushion = 0;

        std::normal_distribution<typename MatrixType::Scalar> normalDistribution{0., 1.};
        DikinEllipsoidCalculator<MatrixType, VectorType> dikinEllipsoidCalculator;
    };

    template<typename InternalMatrixType, typename InternalVectorType>
    DikinProposal<InternalMatrixType, InternalVectorType>::DikinProposal(InternalMatrixType A,
                                                         InternalVectorType b,
                                                         const VectorType& currentState,
                                                         double stepSize) :
            A(std::move(A)),
            b(std::move(b)),
            dikinEllipsoidCalculator(this->A, this->b) {
        DikinProposal::setStepSize(stepSize);
        DikinProposal::setState(currentState);
        proposal = state;
    }

    template<typename InternalMatrixType, typename InternalVectorType>
    std::pair<double, VectorType> DikinProposal<InternalMatrixType, InternalVectorType>::propose(RandomNumberGenerator &randomNumberGenerator) {
        for (long i = 0; i < proposal.rows(); ++i) {
            proposal(i) = normalDistribution(randomNumberGenerator);
        }
        proposal = state + covarianceFactor *
                           stateCholeskyOfDikinEllipsoid.template triangularView<Eigen::Lower>().solve(proposal);


        return {computeLogAcceptanceProbability(), proposal};
    }

    template<typename InternalMatrixType, typename InternalVectorType>
    VectorType DikinProposal<InternalMatrixType, InternalVectorType>::acceptProposal() {
        state.swap(proposal);
        stateCholeskyOfDikinEllipsoid = std::move(proposalCholeskyOfDikinEllipsoid);
        stateLogSqrtDeterminant = proposalLogSqrtDeterminant;
        return state;
    }

    template<typename InternalMatrixType, typename InternalVectorType>
    void DikinProposal<InternalMatrixType, InternalVectorType>::setState(VectorType newState) {
        state.swap(newState);
        auto choleskyResult = dikinEllipsoidCalculator.computeCholeskyFactorOfDikinEllipsoid(state);
        if (!choleskyResult.first) {
            throw std::runtime_error("Could not compute cholesky factorization for newState.");
        }
        stateCholeskyOfDikinEllipsoid = std::move(choleskyResult.second);
        stateLogSqrtDeterminant = stateCholeskyOfDikinEllipsoid.diagonal().array().log().sum();
    }

    template<typename InternalMatrixType, typename InternalVectorType>
    typename InternalMatrixType::Scalar
    DikinProposal<InternalMatrixType, InternalVectorType>::computeLogAcceptanceProbability() {
        bool isProposalInteriorPoint = ((A * proposal - b).array() < -boundaryCushion).all();
        if (!isProposalInteriorPoint) {
            return -std::numeric_limits<typename InternalMatrixType::Scalar>::infinity();
        }

        auto choleskyResult = dikinEllipsoidCalculator.computeCholeskyFactorOfDikinEllipsoid(proposal);
        if (!choleskyResult.first) {
            return -std::numeric_limits<typename InternalMatrixType::Scalar>::infinity();
        }
        proposalCholeskyOfDikinEllipsoid = std::move(choleskyResult.second);

        proposalLogSqrtDeterminant = proposalCholeskyOfDikinEllipsoid.diagonal().array().log().sum();
        InternalVectorType stateDifference = state - proposal;

        return proposalLogSqrtDeterminant
               - stateLogSqrtDeterminant
               + geometricFactor * ((stateCholeskyOfDikinEllipsoid * stateDifference).squaredNorm()
                                    - (proposalCholeskyOfDikinEllipsoid * stateDifference).squaredNorm()
        );
    }

    template<typename InternalMatrixType, typename InternalVectorType>
    void DikinProposal<InternalMatrixType, InternalVectorType>::setStepSize(double newStepSize) {
        stepSize = newStepSize;
        geometricFactor = A.cols() / (2 * stepSize);
        covarianceFactor = std::sqrt(stepSize / A.cols());
    }

    template<typename InternalMatrixType, typename InternalVectorType>
    std::string DikinProposal<InternalMatrixType, InternalVectorType>::getProposalName() const {
        return "DikinWalk";
    }

    template<typename InternalMatrixType, typename InternalVectorType>
    std::optional<double> DikinProposal<InternalMatrixType, InternalVectorType>::getStepSize() const {
        return stepSize;
    }

    template<typename InternalMatrixType, typename InternalVectorType>
    bool DikinProposal<InternalMatrixType, InternalVectorType>::hasStepSize() const {
        return true;
    }

    template<typename InternalMatrixType, typename InternalVectorType>
    std::unique_ptr<Proposal> DikinProposal<InternalMatrixType, InternalVectorType>::deepCopy() const {
        return std::make_unique<DikinProposal>(*this);
    }
}

#endif //HOPS_DIKINPROPOSAL_HPP
