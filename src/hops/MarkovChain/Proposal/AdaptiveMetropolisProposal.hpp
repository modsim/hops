#ifndef HOPS_ADAPTIVEMETROPOLISPROPOSAL_HPP
#define HOPS_ADAPTIVEMETROPOLISPROPOSAL_HPP

#include "IsSetStepSizeAvailable.hpp"
#include <hops/FileWriter/CsvWriter.hpp>
#include <hops/MarkovChain/Proposal/Proposal.hpp>
#include <hops/Polytope/MaximumVolumeEllipsoid.hpp>
#include <hops/RandomNumberGenerator/RandomNumberGenerator.hpp>
#include <hops/Utility/MatrixType.hpp>
#include <hops/Utility/VectorType.hpp>
#include <random>

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
         * @param stepSize          The stepsize by which the trained Gaussian proposal distribution is scaled.
         * @param eps               Scaling factor of the maximum volume ellipsoid, which is added to the covariance to preserve 
         *                          (numerical) irreducibility.
         * @param warmUp            Number of warm up samples during which the maximum volume ellipsoid is used as covariance of 
         *                          the proposal distribution. After the warm up, the adaptive covariance is used.
         */
        AdaptiveMetropolisProposal(InternalMatrixType A, 
                                   InternalVectorType b, 
                                   StateType currentState, 
                                   typename MatrixType::Scalar stepSize = 1, 
                                   typename MatrixType::Scalar eps = 1.e-3, 
                                   unsigned long warmUp = 100);

        std::pair<double, VectorType> propose(RandomNumberGenerator &randomNumberGenerator) override;

        VectorType acceptProposal() override;

        void setState(StateType newState) override;

        [[nodiscard]] VectorType getState() const override;

        VectorType getProposal() const override;

        void setStepSize(double stepSize);

        [[nodiscard]] std::optional<double> getStepSize() const;

        [[nodiscard]] bool hasStepSize() const override;

        [[nodiscard]] std::string getProposalName() const override;

        [[nodiscard]] std::unique_ptr<Proposal> deepCopy() const override;

        [[nodiscard]] double computeLogAcceptanceProbability();

    private:
        MatrixType A;
        VectorType b;
        StateType state;
        StateType proposal;

        StateType stateMean;

        MatrixType stateCovariance;
        MatrixType proposalCovariance;
        MatrixType maximumVolumeEllipsoid;

        MatrixType stateCholeskyOfCovariance;
        MatrixType proposalCholeskyOfCovariance;
        MatrixType choleskyOfMaximumVolumeEllipsoid;

        typename MatrixType::Scalar stateLogSqrtDeterminant;
        typename MatrixType::Scalar proposalLogSqrtDeterminant;

        unsigned long t;
        unsigned long warmUp;

        typename MatrixType::Scalar eps;
        typename MatrixType::Scalar stepSize;
        constexpr static typename MatrixType::Scalar boundaryCushion = 1e-10;

        std::normal_distribution<typename MatrixType::Scalar> normal;

        MatrixType updateCovariance(const MatrixType& covariance, const StateType& mean, const StateType& newState) {
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
                                                                                                   typename MatrixType::Scalar stepSize_,
                                                                                                   typename MatrixType::Scalar eps_,
                                                                                                   unsigned long warmUp_) :
            A(std::move(A_)),
            b(std::move(b_)),
            state(std::move(currentState_)),
            proposal(this->state),
            t(0),
            warmUp(warmUp_),
            stepSize(stepSize_) {
        normal = std::normal_distribution<typename MatrixType::Scalar>(0, stepSize);

        // scale down with larger dimensions according to Roberts & Rosenthal, 2001.
        eps = eps_ / A.cols();

        stateMean = state; // actual content is irrelevant as long as dimensions match

        maximumVolumeEllipsoid = MaximumVolumeEllipsoid<double>::construct(A, b, 10000).getEllipsoid();
        Eigen::LLT<MatrixType> solverMaximumVolumeEllipsoid(maximumVolumeEllipsoid);
        choleskyOfMaximumVolumeEllipsoid = solverMaximumVolumeEllipsoid.matrixL();

        stateCovariance = maximumVolumeEllipsoid;
        Eigen::LLT<MatrixType> solverStateCovariance(stateCovariance);
        stateCholeskyOfCovariance = solverStateCovariance.matrixL();

        stateLogSqrtDeterminant = stateCholeskyOfCovariance.diagonal().array().log().sum(); 
    }

    template<typename InternalMatrixType, typename InternalVectorType>
    std::pair<double, VectorType> AdaptiveMetropolisProposal<InternalMatrixType, InternalVectorType>::propose(
            RandomNumberGenerator &randomNumberGenerator) {
        stateMean = (t * stateMean + state) / (t + 1);

        for (long i = 0; i < proposal.rows(); ++i) {
            proposal(i) = normal(randomNumberGenerator);
        }
        
        if (t > warmUp) {
            proposal = state + stateCholeskyOfCovariance * proposal;
        } else {
            proposal = state + eps * choleskyOfMaximumVolumeEllipsoid.template triangularView<Eigen::Lower>().solve(proposal);
        }
;
        ++t; // increment time
        
        return {computeLogAcceptanceProbability(), proposal};
    }

    template<typename InternalMatrixType, typename InternalVectorType>
    double AdaptiveMetropolisProposal<InternalMatrixType, InternalVectorType>::computeLogAcceptanceProbability() {
        bool isProposalInteriorPoint = ((A * proposal - b).array() < -boundaryCushion).all();
        if (!isProposalInteriorPoint) {
            return -std::numeric_limits<typename MatrixType::Scalar>::infinity();
        }

        proposalCovariance = updateCovariance(stateCovariance, stateMean, proposal);
        Eigen::LLT<Eigen::Matrix<typename MatrixType::Scalar, Eigen::Dynamic, Eigen::Dynamic>> solver(proposalCovariance);
        if (solver.info() != Eigen::Success) {
            return -std::numeric_limits<typename MatrixType::Scalar>::infinity();
        }
        proposalCholeskyOfCovariance = solver.matrixL();

        proposalLogSqrtDeterminant = proposalCholeskyOfCovariance.diagonal().array().log().sum(); 
        StateType stateDifference = proposal - state;

        double alpha = 0;

        // before warm up we have a symmetrical proposal distribution, so we do the next bit only after warm up
        if (t > warmUp) {
            alpha =  stateLogSqrtDeterminant 
                     - proposalLogSqrtDeterminant 
                     - 0.5 * (
                          proposalCholeskyOfCovariance.template triangularView<Eigen::Lower>().solve(stateDifference).squaredNorm() 
                          - stateCholeskyOfCovariance.template triangularView<Eigen::Lower>().solve(stateDifference).squaredNorm()
                     );
        }

        return alpha;
    }

    template<typename InternalMatrixType, typename InternalVectorType>
    VectorType AdaptiveMetropolisProposal<InternalMatrixType, InternalVectorType>::acceptProposal() {
        state.swap(proposal);
        stateCovariance = proposalCovariance;
        stateCholeskyOfCovariance = proposalCholeskyOfCovariance;
        stateLogSqrtDeterminant = proposalLogSqrtDeterminant;
        return state;
    }

    template<typename InternalMatrixType, typename InternalVectorType>
    void AdaptiveMetropolisProposal<InternalMatrixType, InternalVectorType>::setState(VectorType newState) {
        AdaptiveMetropolisProposal::state = std::move(newState);
    }

    template<typename InternalMatrixType, typename InternalVectorType>
    void AdaptiveMetropolisProposal<InternalMatrixType, InternalVectorType>::setStepSize(double newStepSize) {
        stepSize = newStepSize;
        normal = std::normal_distribution<typename MatrixType::Scalar>(0, stepSize);
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
    std::unique_ptr<Proposal> AdaptiveMetropolisProposal<InternalMatrixType, InternalVectorType>::deepCopy() const {
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
}

#endif //HOPS_ADAPTIVEMETROPOLISPROPOSAL_HPP
