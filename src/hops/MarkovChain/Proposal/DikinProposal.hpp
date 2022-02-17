#ifndef HOPS_DIKINPROPOSAL_HPP
#define HOPS_DIKINPROPOSAL_HPP

#include <Eigen/LU>
#include <random>

#include <hops/RandomNumberGenerator/RandomNumberGenerator.hpp>
#include <hops/Utility/MatrixType.hpp>
#include <hops/Utility/StringUtility.hpp>
#include <hops/Utility/VectorType.hpp>

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

        [[nodiscard]] std::vector<std::string> getParameterNames() const override;

        [[nodiscard]] std::any getParameter(const ProposalParameter &parameter) const override;

        [[nodiscard]] std::string getParameterType(const ProposalParameter &parameter) const override;

        void setParameter(const ProposalParameter &parameter, const std::any &value) override;

        void setStepSize(double stepSize);

        [[nodiscard]] std::string getProposalName() const override;

        [[nodiscard]] std::optional<double> getStepSize() const;

        [[nodiscard]] bool hasStepSize() const override;

        [[nodiscard]] std::unique_ptr<Proposal> copyProposal() const override;

        double computeLogAcceptanceProbability() override;

        [[nodiscard]] const MatrixType& getA() const override;

        [[nodiscard]] const VectorType& getB() const override;

        ProposalStatistics & getProposalStatistics() override;

        void activateTrackingOfProposalStatistics() override;

        void disableTrackingOfProposalStatistics() override;

        bool isTrackingOfProposalStatisticsActivated() override;

        ProposalStatistics getAndResetProposalStatistics() override;

    private:
        MatrixType A;
        VectorType b;
        ProposalStatistics proposalStatistics;

        VectorType state;
        VectorType proposal;

        double stateLogSqrtDeterminant = 0;
        double proposalLogSqrtDeterminant = 0;
        MatrixType stateCholeskyOfDikinEllipsoid;
        MatrixType proposalCholeskyOfDikinEllipsoid;

        double stepSize = 0.075; // value from dikin walk publication
        double geometricFactor = 0;
        double covarianceFactor = 0;
        double boundaryCushion = 0;

        std::normal_distribution<double> normalDistribution{0., 1.};
        DikinEllipsoidCalculator <MatrixType, VectorType> dikinEllipsoidCalculator;

        bool isProposalInfosTrackingActive = false;
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
        if (((b - A * newState).array() < 0).any()) {
            throw std::invalid_argument("Starting point outside polytope always gives constant Markov chain.");
        }
        state = newState;
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
        if(isProposalInfosTrackingActive) {
            proposalStatistics.appendInfo("proposal_is_interior", isProposalInteriorPoint);
        }
        if (!isProposalInteriorPoint) {
            if(isProposalInfosTrackingActive) {
                proposalStatistics.appendInfo("proposal_log_sqrt_det", -1);
            }
            return -std::numeric_limits<double>::infinity();
        }

        auto choleskyResult = dikinEllipsoidCalculator.computeCholeskyFactorOfDikinEllipsoid(proposal);
        if (!choleskyResult.first) {
            return -std::numeric_limits<double>::infinity();
        }
        proposalCholeskyOfDikinEllipsoid = std::move(choleskyResult.second);

        proposalLogSqrtDeterminant = proposalCholeskyOfDikinEllipsoid.diagonal().array().log().sum();
        proposalStatistics.appendInfo("proposal_log_sqrt_det", proposalLogSqrtDeterminant);
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
    std::unique_ptr<Proposal> DikinProposal<InternalMatrixType, InternalVectorType>::copyProposal() const {
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
        return {
                ProposalParameterName[static_cast<int>(ProposalParameter::BOUNDARY_CUSHION)],
                ProposalParameterName[static_cast<int>(ProposalParameter::STEP_SIZE)],
        };
    }

    template<typename InternalMatrixType, typename InternalVectorType>
    std::any
    DikinProposal<InternalMatrixType, InternalVectorType>::getParameter(const ProposalParameter &parameter) const {
        if (parameter == ProposalParameter::STEP_SIZE) {
            return std::any(stepSize);
        } else if (parameter == ProposalParameter::BOUNDARY_CUSHION) {
            return std::any(boundaryCushion);
        }
        throw std::invalid_argument("Can't get parameter which doesn't exist in " + this->getProposalName());
    }

    template<typename InternalMatrixType, typename InternalVectorType>
    std::string
    DikinProposal<InternalMatrixType, InternalVectorType>::getParameterType(const ProposalParameter &parameter) const {
        if (parameter == ProposalParameter::STEP_SIZE) {
            return "double";
        } else if (parameter == ProposalParameter::BOUNDARY_CUSHION) {
            return "double";
        } else {
            throw std::invalid_argument("Can't get parameter which doesn't exist in " + this->getProposalName());
        }
    }

    template<typename InternalMatrixType, typename InternalVectorType>
    void DikinProposal<InternalMatrixType, InternalVectorType>::setParameter(const ProposalParameter &parameter,
                                                                             const std::any &value) {
        if (parameter == ProposalParameter::STEP_SIZE) {
            setStepSize(std::any_cast<double>(value));
        } else if (parameter == ProposalParameter::BOUNDARY_CUSHION) {
            boundaryCushion = std::any_cast<double>(value);
        } else {
            throw std::invalid_argument("Can't get parameter which doesn't exist in " + this->getProposalName());
        }

    }

    template<typename InternalMatrixType, typename InternalVectorType>
    const MatrixType& DikinProposal<InternalMatrixType, InternalVectorType>::getA() const {
        return A;
    }

    template<typename InternalMatrixType, typename InternalVectorType>
    const VectorType& DikinProposal<InternalMatrixType, InternalVectorType>::getB() const {
        return b;
    }

    template<typename InternalMatrixType, typename InternalVectorType>
    ProposalStatistics & DikinProposal<InternalMatrixType, InternalVectorType>::getProposalStatistics() {
        return proposalStatistics;
    }

    template<typename InternalMatrixType, typename InternalVectorType>
    void DikinProposal<InternalMatrixType, InternalVectorType>::activateTrackingOfProposalStatistics() {
        isProposalInfosTrackingActive = true;
    }

    template<typename InternalMatrixType, typename InternalVectorType>
    void DikinProposal<InternalMatrixType, InternalVectorType>::disableTrackingOfProposalStatistics() {
        isProposalInfosTrackingActive = false;
    }

    template<typename InternalMatrixType, typename InternalVectorType>
    bool DikinProposal<InternalMatrixType, InternalVectorType>::isTrackingOfProposalStatisticsActivated() {
        return isProposalInfosTrackingActive;
    }

    template<typename InternalMatrixType, typename InternalVectorType>
    ProposalStatistics DikinProposal<InternalMatrixType, InternalVectorType>::getAndResetProposalStatistics() {
        ProposalStatistics newStatistic;
        std::swap(newStatistic, proposalStatistics);
        return newStatistic;
    }
}

#endif //HOPS_DIKINPROPOSAL_HPP
