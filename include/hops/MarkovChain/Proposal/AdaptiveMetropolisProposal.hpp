#ifndef HOPS_ADAPTIVEMETROPOLISPROPOSAL_HPP
#define HOPS_ADAPTIVEMETROPOLISPROPOSAL_HPP

#include "../IsSetStepSizeAvailable.hpp"
#include "../../RandomNumberGenerator/RandomNumberGenerator.hpp"
#include "../../FileWriter/CsvWriter.hpp"
#include "../../Polytope/MaximumVolumeEllipsoid.hpp"
#include <random>

namespace hops {
    template<typename MatrixType, typename VectorType>
    class AdaptiveMetropolisProposal {
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
        AdaptiveMetropolisProposal(MatrixType A, 
                                   VectorType b, 
                                   StateType currentState, 
                                   typename MatrixType::Scalar stepSize = 1, 
                                   typename MatrixType::Scalar eps = 1.e-3, 
                                   unsigned long warmUp = 100);

        void propose(RandomNumberGenerator &randomNumberGenerator);

        void acceptProposal();

        StateType getState() const;

        StateType getProposal() const;

        void setState(StateType newState);

        void setStepSize(typename MatrixType::Scalar stepSize);

        typename MatrixType::Scalar getStepSize() const;

        [[nodiscard]] typename MatrixType::Scalar calculateLogAcceptanceProbability(); 

        std::string getName();

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

    template<typename MatrixType, typename VectorType>
    AdaptiveMetropolisProposal<MatrixType, VectorType>::AdaptiveMetropolisProposal(MatrixType A_,
                                                               VectorType b_,
                                                               VectorType currentState_,
                                                               typename MatrixType::Scalar stepSize_,
                                                               typename MatrixType::Scalar eps_,
                                                               unsigned long warmUp_) :
            A(std::move(A_)),
            b(std::move(b_)),
            state(std::move(currentState_)),
            proposal(this->state),
            stepSize(stepSize_),
            warmUp(warmUp_),
            t(0) {
        normal = std::normal_distribution<typename MatrixType::Scalar>(0, stepSize);

        // scale down with larger dimensions according to Roberts & Rosenthal, 2001.
        eps = eps_ / A.cols();

        stateMean = state; // actual content is irrelevant as long as dimensions match

        maximumVolumeEllipsoid = MaximumVolumeEllipsoid<double>::construct(A, b, 10000).getEllipsoid();
        Eigen::LLT<Eigen::Matrix<typename MatrixType::Scalar, Eigen::Dynamic, Eigen::Dynamic>> solverMaximumVolumeEllipsoid(maximumVolumeEllipsoid);
        choleskyOfMaximumVolumeEllipsoid = solverMaximumVolumeEllipsoid.matrixL();

        stateCovariance = maximumVolumeEllipsoid;
        Eigen::LLT<Eigen::Matrix<typename MatrixType::Scalar, Eigen::Dynamic, Eigen::Dynamic>> solverStateCovariance(stateCovariance);
        stateCholeskyOfCovariance = solverStateCovariance.matrixL();

        stateLogSqrtDeterminant = stateCholeskyOfCovariance.diagonal().array().log().sum(); 
    }

    template<typename MatrixType, typename VectorType>
    void AdaptiveMetropolisProposal<MatrixType, VectorType>::propose(
            RandomNumberGenerator &randomNumberGenerator) {
        stateMean = (t * stateMean + state) / (t + 1);

        for (long i = 0; i < proposal.rows(); ++i) {
            proposal(i) = normal(randomNumberGenerator);
        }
        
        if (t > warmUp) {
            proposal = state + stateCholeskyOfCovariance.template triangularView<Eigen::Lower>().solve(proposal);
        } else {
            proposal = state + 0.01 * choleskyOfMaximumVolumeEllipsoid.template triangularView<Eigen::Lower>().solve(proposal);
        }
;
        ++t; // increment time
    }

    template<typename MatrixType, typename VectorType>
    void
    AdaptiveMetropolisProposal<MatrixType, VectorType>::acceptProposal() {
        state.swap(proposal);
        stateCovariance = proposalCovariance;
        stateCholeskyOfCovariance = proposalCholeskyOfCovariance;
        stateLogSqrtDeterminant = proposalLogSqrtDeterminant;
    }

    template<typename MatrixType, typename VectorType>
    typename MatrixType::Scalar 
    AdaptiveMetropolisProposal<MatrixType, VectorType>::calculateLogAcceptanceProbability() {
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

        // before warm up we have a symmetrical proposal distribution
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

    template<typename MatrixType, typename VectorType>
    typename AdaptiveMetropolisProposal<MatrixType, VectorType>::StateType
    AdaptiveMetropolisProposal<MatrixType, VectorType>::getState() const {
        return state;
    }

    template<typename MatrixType, typename VectorType>
    typename AdaptiveMetropolisProposal<MatrixType, VectorType>::StateType
    AdaptiveMetropolisProposal<MatrixType, VectorType>::getProposal() const {
        return proposal;
    }

    template<typename MatrixType, typename VectorType>
    void AdaptiveMetropolisProposal<MatrixType, VectorType>::setState(VectorType newState) {
        AdaptiveMetropolisProposal::state = std::move(newState);
    }

    template<typename MatrixType, typename VectorType>
    void AdaptiveMetropolisProposal<MatrixType, VectorType>::setStepSize(
            typename MatrixType::Scalar newStepSize) {
        stepSize = newStepSize;
        normal = std::normal_distribution<typename MatrixType::Scalar>(0, stepSize);
    }

    template<typename MatrixType, typename VectorType>
    typename MatrixType::Scalar
    AdaptiveMetropolisProposal<MatrixType, VectorType>::getStepSize() const {
        return stepSize;
    }

    template<typename MatrixType, typename VectorType>
    std::string AdaptiveMetropolisProposal<MatrixType, VectorType>::getName() {
        return "AdaptiveMetropolis";
    }
}

#endif //HOPS_ADAPTIVEMETROPOLISPROPOSAL_HPP
