#ifndef HOPS_ADAPTIVEMETROPOLISPROPOSAL_HPP
#define HOPS_ADAPTIVEMETROPOLISPROPOSAL_HPP

#include <random>

#include <hops/MarkovChain/Proposal/Proposal.hpp>
#include <hops/Polytope/MaximumVolumeEllipsoid.hpp>
#include <hops/RandomNumberGenerator/RandomNumberGenerator.hpp>
#include <hops/Utility/MatrixType.hpp>
#include <hops/Utility/StringUtility.hpp>
#include <hops/Utility/VectorType.hpp>

namespace hops {
    template<typename InternalMatrixType = MatrixType, typename InternalVectorType = VectorType>
    class AdaptiveMetropolisProposal : public Proposal {
    public:
        using StateType = VectorType;

        /**
         * @brief Constructs the Adaptive Metropolis proposal mechanism (Haario et al. 2001) on polytope defined as Ax<b.
         * @param A
         * @param b
         * @param currentState
         * @param stepSize          The step size by which the trained Gaussian proposal distribution is scaled.
         * @param eps               Scaling factor of the maximum volume ellipsoid, which is added to the covariance to preserve 
         *                          (numerical) irreducibility.
         * @param warmUp            Number of warm up samples during which the maximum volume ellipsoid is used as covariance of 
         *                          the proposal distribution. After the warm up, the adaptive covariance is used.
         */
        AdaptiveMetropolisProposal(InternalMatrixType A,
                                   InternalVectorType b,
                                   StateType currentState,
                                   double stepSize = 1,
                                   double eps = 1.e-3,
                                   unsigned long warmUp = 100,
                                   unsigned long t = 0);

        AdaptiveMetropolisProposal(InternalMatrixType A,
                                   InternalVectorType b,
                                   StateType currentState,
                                   const MatrixType& sqrtMaximumVolumeEllipsoid,
                                   double stepSize = 1,
                                   double eps = 1.e-3,
                                   unsigned long warmUp = 100,
                                   unsigned long t = 0);

        VectorType &propose(RandomNumberGenerator &randomNumberGenerator) override;

        VectorType &acceptProposal() override;

        void setState(const StateType &newState) override;

        [[nodiscard]] VectorType getState() const override;

        [[nodiscard]] VectorType getProposal() const override;

        [[nodiscard]] std::vector<std::string> getParameterNames() const override;

        [[nodiscard]] std::any getParameter(const ProposalParameter &parameter) const override;

        [[nodiscard]] std::string getParameterType(const ProposalParameter &parameter) const override;

        void setParameter(const ProposalParameter &parameter, const std::any &value) override;

        void setStepSize(double stepSize);

        [[nodiscard]] std::optional<double> getStepSize() const;

        [[nodiscard]] bool hasStepSize() const override;

        [[nodiscard]] std::string getProposalName() const override;

        [[nodiscard]] std::unique_ptr<Proposal> copyProposal() const override;

        [[nodiscard]] double computeLogAcceptanceProbability() override;

        [[nodiscard]] const MatrixType& getA() const override;

        [[nodiscard]] const VectorType& getB() const override;

        [[nodiscard]] const MatrixType& getCholeskyOfMaximumVolumeEllipsoid() const;

        [[nodiscard]] unsigned long getT() const;

        ProposalStatistics & getProposalStatistics() override;

        void activateTrackingOfProposalStatistics() override;

        void disableTrackingOfProposalStatistics() override;

        bool isTrackingOfProposalStatisticsActivated() override;

        ProposalStatistics getAndResetProposalStatistics() override;

    protected:
        // These protected types are/should be accessed in BillliardAdaptiveMetropolisProposal only
        MatrixType A;
        VectorType b;

    private:
        StateType state;
        StateType proposal;
        ProposalStatistics proposalStatistics;

        StateType stateMean;

        MatrixType stateCovariance;
        MatrixType proposalCovariance;
        MatrixType maximumVolumeEllipsoid;

        MatrixType stateCholeskyOfCovariance;
        MatrixType proposalCholeskyOfCovariance;
        MatrixType choleskyOfMaximumVolumeEllipsoid;

        double stateLogSqrtDeterminant;
        double proposalLogSqrtDeterminant;

        unsigned long t;
        unsigned long warmUp;

        double eps;
        double stepSize;
        double boundaryCushion = 0; 

        std::normal_distribution<double> normal;

        bool isProposalInfosTrackingActive = false;

        MatrixType updateCovariance(const MatrixType &covariance, const StateType &mean, const StateType &newState) {
            assert(t > 0 && "cannot update covariance without samples having been drawn");

            // recursive mean
            StateType newMean = (t * mean + newState) / (t + 1);
            MatrixType newCovariance = ((t - 1) * covariance
                                        + t * (mean * mean.transpose())
                                        - (t + 1) * (newMean * newMean.transpose())
                                        + newState * newState.transpose()
                                        + eps * maximumVolumeEllipsoid) / t;
            return newCovariance;
        }
    };

    template<typename InternalMatrixType, typename InternalVectorType>
    AdaptiveMetropolisProposal<InternalMatrixType, InternalVectorType>::AdaptiveMetropolisProposal(InternalMatrixType A_,
                                                                                                   InternalVectorType b_,
                                                                                                   VectorType currentState_,
                                                                                                   const MatrixType& sqrtMaximumVolumeEllipsoid,
                                                                                                   double stepSize_,
                                                                                                   double eps_,
                                                                                                   unsigned long warmUp_,
                                                                                                   unsigned long t_) :
            A(std::move(A_)),
            b(std::move(b_)),
            state(std::move(currentState_)),
            proposal(this->state),
            t(t_),
            warmUp(warmUp_),
            stepSize(stepSize_) {
        normal = std::normal_distribution<double>(0, stepSize);

        // scale down with larger dimensions according to Roberts & Rosenthal, 2001.
        eps = eps_ / A.cols();

        stateMean = state; // actual content is irrelevant as long as dimensions match

        this->maximumVolumeEllipsoid = sqrtMaximumVolumeEllipsoid * sqrtMaximumVolumeEllipsoid.transpose();
        stateCovariance = this->maximumVolumeEllipsoid;
        choleskyOfMaximumVolumeEllipsoid = sqrtMaximumVolumeEllipsoid;
        stateCholeskyOfCovariance = sqrtMaximumVolumeEllipsoid;

        stateLogSqrtDeterminant = stateCholeskyOfCovariance.diagonal().array().log().sum();
        proposalLogSqrtDeterminant = stateLogSqrtDeterminant;
        proposalCovariance = stateCovariance;
    }

    template<typename InternalMatrixType, typename InternalVectorType>
    AdaptiveMetropolisProposal<InternalMatrixType, InternalVectorType>::AdaptiveMetropolisProposal(InternalMatrixType A_,
                                                                                                   InternalVectorType b_,
                                                                                                   VectorType currentState_,
                                                                                                   double stepSize_,
                                                                                                   double eps_,
                                                                                                   unsigned long warmUp_,
                                                                                                   unsigned long t_) :
            A(std::move(A_)),
            b(std::move(b_)),
            state(std::move(currentState_)),
            proposal(this->state),
            t(t_),
            warmUp(warmUp_),
            stepSize(stepSize_) {
        normal = std::normal_distribution<double>(0, stepSize);

        // scale down with larger dimensions according to Roberts & Rosenthal, 2001.
        eps = eps_ / A.cols();

        stateMean = state; // actual content is irrelevant as long as dimensions match

        auto MVE = MaximumVolumeEllipsoid<double>::construct(A, b, 10000);
        maximumVolumeEllipsoid = MVE.getEllipsoid();
        stateCovariance = maximumVolumeEllipsoid;
        Eigen::LLT<MatrixType> solverMaximumVolumeEllipsoid(maximumVolumeEllipsoid);
        choleskyOfMaximumVolumeEllipsoid = MVE.getRoundingTransformation();
        stateCholeskyOfCovariance = MVE.getRoundingTransformation();

        stateLogSqrtDeterminant = stateCholeskyOfCovariance.diagonal().array().log().sum();
        proposalLogSqrtDeterminant = stateLogSqrtDeterminant;
        proposalCovariance = stateCovariance;
    }

    template<typename InternalMatrixType, typename InternalVectorType>
    VectorType &AdaptiveMetropolisProposal<InternalMatrixType, InternalVectorType>::propose(
            RandomNumberGenerator &randomNumberGenerator) {
        stateMean = (t * stateMean + state) / (t + 1);

        for (long i = 0; i < proposal.rows(); ++i) {
            proposal(i) = normal(randomNumberGenerator);
        }

        if (t > warmUp) {
            proposal = state + stateCholeskyOfCovariance * proposal;
        } else {
            proposal = state +
                       eps * choleskyOfMaximumVolumeEllipsoid.template triangularView<Eigen::Lower>().solve(proposal);
        };
        ++t; // increment time

        return proposal;
    }

    template<typename InternalMatrixType, typename InternalVectorType>
    double AdaptiveMetropolisProposal<InternalMatrixType, InternalVectorType>::computeLogAcceptanceProbability() {
        bool isProposalInteriorPoint = ((A * proposal - b).array() < -boundaryCushion).all();
        if (!isProposalInteriorPoint) {
            return -std::numeric_limits<double>::infinity();
        }

        proposalCovariance = updateCovariance(stateCovariance, stateMean, proposal);
        Eigen::LLT<MatrixType> solver(
                proposalCovariance);
        if (solver.info() != Eigen::Success) {
            return -std::numeric_limits<double>::infinity();
        }
        proposalCholeskyOfCovariance = solver.matrixL();

        proposalLogSqrtDeterminant = proposalCholeskyOfCovariance.diagonal().array().log().sum();
        StateType stateDifference = proposal - state;

        double alpha = 0;

        // before warm up we have a symmetrical proposal distribution, so we do the next bit only after warm up
        if (t > warmUp) {
            alpha = stateLogSqrtDeterminant
                    - proposalLogSqrtDeterminant
                    - 0.5 * (
                    proposalCholeskyOfCovariance.template triangularView<Eigen::Lower>().solve(
                            stateDifference).squaredNorm()
                    - stateCholeskyOfCovariance.template triangularView<Eigen::Lower>().solve(
                            stateDifference).squaredNorm()
            );
        }

        return alpha;
    }

    template<typename InternalMatrixType, typename InternalVectorType>
    VectorType &AdaptiveMetropolisProposal<InternalMatrixType, InternalVectorType>::acceptProposal() {
        state.swap(proposal);
        stateCovariance = proposalCovariance;
        stateCholeskyOfCovariance = proposalCholeskyOfCovariance;
        stateLogSqrtDeterminant = proposalLogSqrtDeterminant;
        return state;
    }

    template<typename InternalMatrixType, typename InternalVectorType>
    void AdaptiveMetropolisProposal<InternalMatrixType, InternalVectorType>::setState(const VectorType &newState) {
        AdaptiveMetropolisProposal::state = newState;
    }

    template<typename InternalMatrixType, typename InternalVectorType>
    void AdaptiveMetropolisProposal<InternalMatrixType, InternalVectorType>::setStepSize(double newStepSize) {
        stepSize = newStepSize;
        normal = std::normal_distribution<double>(0, stepSize);
    }


    template<typename InternalMatrixType, typename InternalVectorType>
    std::optional<double> AdaptiveMetropolisProposal<InternalMatrixType, InternalVectorType>::getStepSize() const {
        return stepSize;
    }

    template<typename InternalMatrixType, typename InternalVectorType>
    bool AdaptiveMetropolisProposal<InternalMatrixType, InternalVectorType>::hasStepSize() const {
        return true;
    }

    template<typename InternalMatrixType, typename InternalVectorType>
    std::string AdaptiveMetropolisProposal<InternalMatrixType, InternalVectorType>::getProposalName() const {
        return "AdaptiveMetropolis";
    }

    template<typename InternalMatrixType, typename InternalVectorType>
    std::unique_ptr<Proposal> AdaptiveMetropolisProposal<InternalMatrixType, InternalVectorType>::copyProposal() const {
        return std::make_unique<AdaptiveMetropolisProposal>(*this);
    }

    template<typename InternalMatrixType, typename InternalVectorType>
    VectorType AdaptiveMetropolisProposal<InternalMatrixType, InternalVectorType>::getState() const {
        return state;
    }

    template<typename InternalMatrixType, typename InternalVectorType>
    VectorType AdaptiveMetropolisProposal<InternalMatrixType, InternalVectorType>::getProposal() const {
        return proposal;
    }

    template<typename InternalMatrixType, typename InternalVectorType>
    void AdaptiveMetropolisProposal<InternalMatrixType, InternalVectorType>::setParameter(
            const ProposalParameter &parameter, const std::any &value) {
        if (parameter == ProposalParameter::BOUNDARY_CUSHION) {
            this->boundaryCushion = std::any_cast<double>(value);
        } else if (parameter == ProposalParameter::EPSILON) {
            this->eps = std::any_cast<double>(value);
        } else if (parameter == ProposalParameter::STEP_SIZE) {
            setStepSize(std::any_cast<double>(value));
        } else if (parameter == ProposalParameter::WARM_UP) {
            this->warmUp = std::any_cast<long>(value);
        } else {
            throw std::invalid_argument("Can't get parameter which doesn't exist in " + this->getProposalName());
        }
    }

    template<typename InternalMatrixType, typename InternalVectorType>
    std::vector<std::string>
    AdaptiveMetropolisProposal<InternalMatrixType, InternalVectorType>::getParameterNames() const {
        return {
            ProposalParameterName[static_cast<int>(ProposalParameter::BOUNDARY_CUSHION)],
            ProposalParameterName[static_cast<int>(ProposalParameter::EPSILON)],
            ProposalParameterName[static_cast<int>(ProposalParameter::STEP_SIZE)],
            ProposalParameterName[static_cast<int>(ProposalParameter::WARM_UP)],
        };
    }

    template<typename InternalMatrixType, typename InternalVectorType>
    std::any AdaptiveMetropolisProposal<InternalMatrixType, InternalVectorType>::getParameter(
            const ProposalParameter &parameter) const {
        if (parameter == ProposalParameter::BOUNDARY_CUSHION) {
            return std::any(boundaryCushion);
        } else if (parameter == ProposalParameter::EPSILON) {
            return std::any(eps);
        } else if (parameter == ProposalParameter::STEP_SIZE) {
            return std::any(stepSize);
        } else if (parameter == ProposalParameter::WARM_UP) {
            return std::any(warmUp);
        } else {
            throw std::invalid_argument("Can't get parameter which doesn't exist in " + this->getProposalName());
        }
    }

    template<typename InternalMatrixType, typename InternalVectorType>
    std::string AdaptiveMetropolisProposal<InternalMatrixType, InternalVectorType>::getParameterType(
            const ProposalParameter &parameter) const {
        if (parameter == ProposalParameter::BOUNDARY_CUSHION) {
            return "double";
        } else if (parameter == ProposalParameter::EPSILON) {
            return "double";
        } else if (parameter == ProposalParameter::STEP_SIZE) {
            return "double";
        } else if (parameter == ProposalParameter::WARM_UP) {
            return "long";
        } else {
            throw std::invalid_argument("Can't get parameter which doesn't exist in " + this->getProposalName());
        }
    }

    template<typename InternalMatrixType, typename InternalVectorType>
    const MatrixType& AdaptiveMetropolisProposal<InternalMatrixType, InternalVectorType>::getA() const {
        return A;
    }

    template<typename InternalMatrixType, typename InternalVectorType>
    const VectorType& AdaptiveMetropolisProposal<InternalMatrixType, InternalVectorType>::getB() const {
        return b;
    }

    template<typename InternalMatrixType, typename InternalVectorType>
    const MatrixType& AdaptiveMetropolisProposal<InternalMatrixType, InternalVectorType>::getCholeskyOfMaximumVolumeEllipsoid() const {
        return choleskyOfMaximumVolumeEllipsoid;
    }

    template<typename InternalMatrixType, typename InternalVectorType>
    unsigned long AdaptiveMetropolisProposal<InternalMatrixType, InternalVectorType>::getT() const {
        return t;
    }

    template<typename InternalMatrixType, typename InternalVectorType>
    ProposalStatistics & AdaptiveMetropolisProposal<InternalMatrixType, InternalVectorType>::getProposalStatistics() {
        return proposalStatistics;
    }

    template<typename InternalMatrixType, typename InternalVectorType>
    void AdaptiveMetropolisProposal<InternalMatrixType, InternalVectorType>::activateTrackingOfProposalStatistics() {
        isProposalInfosTrackingActive = true;
    }

    template<typename InternalMatrixType, typename InternalVectorType>
    void AdaptiveMetropolisProposal<InternalMatrixType, InternalVectorType>::disableTrackingOfProposalStatistics() {
        isProposalInfosTrackingActive = false;
    }

    template<typename InternalMatrixType, typename InternalVectorType>
    bool AdaptiveMetropolisProposal<InternalMatrixType, InternalVectorType>::isTrackingOfProposalStatisticsActivated() {
        return isProposalInfosTrackingActive;
    }

    template<typename InternalMatrixType, typename InternalVectorType>
    ProposalStatistics
    AdaptiveMetropolisProposal<InternalMatrixType, InternalVectorType>::getAndResetProposalStatistics() {
        ProposalStatistics newStatistic;
        std::swap(newStatistic, proposalStatistics);
        return newStatistic;
    }
}

#endif //HOPS_ADAPTIVEMETROPOLISPROPOSAL_HPP
