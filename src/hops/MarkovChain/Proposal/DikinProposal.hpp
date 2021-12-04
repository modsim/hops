#ifndef HOPS_DIKINPROPOSAL_HPP
#define HOPS_DIKINPROPOSAL_HPP

#include <Eigen/LU>
#include <random>

#include <hops/RandomNumberGenerator/RandomNumberGenerator.hpp>
#include <hops/Utility/StringUtility.hpp>

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
        DikinProposal(InternalMatrixType A, InternalVectorType b, const VectorType &currentState,
                      double stepSize = 0.075);

        VectorType &propose(RandomNumberGenerator &randomNumberGenerator) override;

        VectorType &acceptProposal() override;

        void setState(const VectorType &newState) override;

        [[nodiscard]] VectorType getState() const override;

        [[nodiscard]] VectorType getProposal() const override;

        std::vector<std::string> getDimensionNames() const override;

        [[nodiscard]] std::vector<std::string> getParameterNames() const override;

        [[nodiscard]] std::any getParameter(const std::string &parameterName) const override;

        [[nodiscard]] std::string getParameterType(const std::string &name) const override;

        void setParameter(const std::string &parameterName, const std::any &value) override;

        void setStepSize(double stepSize);

        [[nodiscard]] std::string getProposalName() const override;

        [[nodiscard]] std::optional<double> getStepSize() const;

        [[nodiscard]] bool hasStepSize() const override;

        [[nodiscard]] std::unique_ptr<Proposal> deepCopy() const override;

        double computeLogAcceptanceProbability() override;

    private:
        MatrixType A;
        VectorType b;

        VectorType state;
        VectorType proposal;

        typename MatrixType::Scalar stateLogSqrtDeterminant = 0;
        typename MatrixType::Scalar proposalLogSqrtDeterminant = 0;
        Eigen::Matrix<typename MatrixType::Scalar, Eigen::Dynamic, Eigen::Dynamic> stateCholeskyOfDikinEllipsoid;
        Eigen::Matrix<typename MatrixType::Scalar, Eigen::Dynamic, Eigen::Dynamic> proposalCholeskyOfDikinEllipsoid;

        typename MatrixType::Scalar stepSize = 0.075; // value from dikin walk publication
        typename MatrixType::Scalar geometricFactor = 0;
        typename MatrixType::Scalar covarianceFactor = 0;
        constexpr static typename MatrixType::Scalar boundaryCushion = 0;

        std::normal_distribution<typename MatrixType::Scalar> normalDistribution{0., 1.};
        DikinEllipsoidCalculator<MatrixType, VectorType> dikinEllipsoidCalculator;

    };

    template<typename InternalMatrixType, typename InternalVectorType>
    DikinProposal<InternalMatrixType, InternalVectorType>::DikinProposal(InternalMatrixType A,
                                                                         InternalVectorType b,
                                                                         const VectorType &currentState,
                                                                         double stepSize) :
            A(std::move(A)),
            b(std::move(b)),
            dikinEllipsoidCalculator(this->A, this->b) {
        DikinProposal::setStepSize(stepSize);
        DikinProposal::setState(currentState);
        proposal = state;
    }

    template<typename InternalMatrixType, typename InternalVectorType>
    VectorType &
    DikinProposal<InternalMatrixType, InternalVectorType>::propose(RandomNumberGenerator &randomNumberGenerator) {
        for (long i = 0; i < proposal.rows(); ++i) {
            proposal(i) = normalDistribution(randomNumberGenerator);
        }
        proposal = state + covarianceFactor *
                           stateCholeskyOfDikinEllipsoid.template triangularView<Eigen::Lower>().solve(proposal);


        return proposal;
    }

    template<typename InternalMatrixType, typename InternalVectorType>
    VectorType &DikinProposal<InternalMatrixType, InternalVectorType>::acceptProposal() {
        state.swap(proposal);
        stateCholeskyOfDikinEllipsoid = std::move(proposalCholeskyOfDikinEllipsoid);
        stateLogSqrtDeterminant = proposalLogSqrtDeterminant;
        return state;
    }

    template<typename InternalMatrixType, typename InternalVectorType>
    void DikinProposal<InternalMatrixType, InternalVectorType>::setState(const VectorType &newState) {
        state.swap(newState);
        auto choleskyResult = dikinEllipsoidCalculator.computeCholeskyFactorOfDikinEllipsoid(state);
        if (!choleskyResult.first) {
            throw std::runtime_error("Could not compute cholesky factorization for newState.");
        }
        stateCholeskyOfDikinEllipsoid = std::move(choleskyResult.second);
        stateLogSqrtDeterminant = stateCholeskyOfDikinEllipsoid.diagonal().array().log().sum();
    }

    template<typename InternalMatrixType, typename InternalVectorType>
    double DikinProposal<InternalMatrixType, InternalVectorType>::computeLogAcceptanceProbability() {
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

    template<typename InternalMatrixType, typename InternalVectorType>
    VectorType DikinProposal<InternalMatrixType, InternalVectorType>::getState() const {
        return state;
    }

    template<typename InternalMatrixType, typename InternalVectorType>
    VectorType DikinProposal<InternalMatrixType, InternalVectorType>::getProposal() const {
        return proposal;
    }

    template<typename InternalMatrixType, typename InternalVectorType>
    std::vector<std::string> DikinProposal<InternalMatrixType, InternalVectorType>::getParameterNames() const {
        return {"step_size"};
    }

    template<typename InternalMatrixType, typename InternalVectorType>
    std::any
    DikinProposal<InternalMatrixType, InternalVectorType>::getParameter(const std::string &parameterName) const {
        std::string lowerCaseParameterName = toLowerCase(parameterName);
        if (lowerCaseParameterName == "step_size") {
            return std::any(stepSize);
        }
        throw std::invalid_argument("Can't get parameter which doesn't exist in " + this->getProposalName());
    }

    template<typename InternalMatrixType, typename InternalVectorType>
    std::string DikinProposal<InternalMatrixType, InternalVectorType>::getParameterType(const std::string &name) const {
        std::string lowerCaseParameterName = toLowerCase(name);
        if (lowerCaseParameterName == "step_size") {
            return "double";
        } else {
            throw std::invalid_argument("Can't get parameter which doesn't exist in " + this->getProposalName());
        }
    }

    template<typename InternalMatrixType, typename InternalVectorType>
    void DikinProposal<InternalMatrixType, InternalVectorType>::setParameter(const std::string &parameterName,
                                                                             const std::any &value) {
        std::string lowerCaseParameterName = toLowerCase(parameterName);
        if (lowerCaseParameterName == "step_size") {
            setStepSize(std::any_cast<double>(value));
        } else {
            throw std::invalid_argument("Can't get parameter which doesn't exist in " + this->getProposalName());
        }

    }

    template<typename InternalMatrixType, typename InternalVectorType>
    std::vector<std::string> DikinProposal<InternalMatrixType, InternalVectorType>::getDimensionNames() const {
        std::vector<std::string> names;
        for (long i = 0; i < state.rows(); ++i) {
            names.emplace_back("x_" + std::to_string(i));
        }
        return names;
    }

}

#endif //HOPS_DIKINPROPOSAL_HPP
