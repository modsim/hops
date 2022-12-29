#ifndef HOPS_ADAPTIVEMETROPOLISPROPOSAL_HPP
#define HOPS_ADAPTIVEMETROPOLISPROPOSAL_HPP

#include <optional>
#include <random>
#include <stdexcept>

#include "hops/MarkovChain/Proposal/Proposal.hpp"
#include "hops/Polytope/MaximumVolumeEllipsoid.hpp"
#include "hops/RandomNumberGenerator/RandomNumberGenerator.hpp"
#include "hops/Utility/DefaultDimensionNames.hpp"
#include "hops/Utility/MatrixType.hpp"
#include "hops/Utility/StringUtility.hpp"
#include "hops/Utility/VectorType.hpp"

namespace hops {
    template<typename InternalMatrixType = MatrixType>
    class AdaptiveMetropolisProposal : public Proposal {
    public:
        /**
         * @brief Constructs the Adaptive Metropolis proposal mechanism (Haario et al. 2001) on polytope defined as Ax<b.
         * @param A
         * @param b
         * @param currentState
         * @param eps               Scaling factor of the maximum volume ellipsoid, which is added to the covariance to preserve
         *                          (numerical) irreducibility.
         * @param warmUp            Number of warm up samples during which the maximum volume ellipsoid is used as covariance of 
         *                          the proposal distribution. After the warm up, the adaptive covariance is used.
         */
        AdaptiveMetropolisProposal(InternalMatrixType A,
                                   VectorType b,
                                   VectorType currentState,
                                   double eps = 1.e-3,
                                   unsigned long warmUp = 100,
                                   unsigned long t = 0);

        AdaptiveMetropolisProposal(InternalMatrixType A,
                                   VectorType b,
                                   VectorType currentState,
                                   const MatrixType &sqrtMaximumVolumeEllipsoid,
                                   double eps = 1.e-3,
                                   unsigned long warmUp = 100,
                                   unsigned long t = 0);

        VectorType &propose(RandomNumberGenerator &randomNumberGenerator) override;

        VectorType &propose(RandomNumberGenerator &rng, const Eigen::VectorXd &activeIndices) override;

        VectorType &acceptProposal() override;

        void setState(const VectorType &newState) override;

        void setProposal(const VectorType &newProposal) override;

        [[nodiscard]] VectorType getState() const override;

        [[nodiscard]] VectorType getProposal() const override;

        void setDimensionNames(const std::vector<std::string> &names) override;

        [[nodiscard]] std::vector<std::string> getDimensionNames() const override;

        [[nodiscard]] std::vector<std::string> getParameterNames() const override;

        [[nodiscard]] std::any getParameter(const ProposalParameter &parameter) const override;

        [[nodiscard]] std::string getParameterType(const ProposalParameter &parameter) const override;

        void setParameter(const ProposalParameter &parameter, const std::any &value) override;

        [[nodiscard]] std::optional<double> getStepSize() const override;

        [[nodiscard]] static bool hasStepSize();

        [[nodiscard]] std::string getProposalName() const override;

        [[nodiscard]] std::unique_ptr<Proposal> copyProposal() const override;

        [[nodiscard]] double computeLogAcceptanceProbability() override;

        [[nodiscard]] const MatrixType &getA() const override;

        [[nodiscard]] const VectorType &getB() const override;

        [[nodiscard]] const MatrixType &getCholeskyOfMaximumVolumeEllipsoid() const;

        [[nodiscard]] unsigned long getT() const;

    protected:
        // These protected types are/should be accessed in BillliardAdaptiveMetropolisProposal only
        MatrixType A;
        VectorType b;

    private:
        VectorType state;
        VectorType proposal;

        VectorType stateMean;

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
        const constexpr static double stepSize = 1;
        double boundaryCushion = 0;

        std::vector<std::string> dimensionNames;

        std::normal_distribution<double> normal;

        MatrixType updateCovariance(const MatrixType &covariance, const VectorType &mean, const VectorType &newState) {
            assert(t > 0 && "cannot update covariance without samples having been drawn");

            // recursive mean
            VectorType newMean = (t * mean + newState) / (t + 1);
            MatrixType newCovariance = ((t - 1) * covariance
                                        + t * (mean * mean.transpose())
                                        - (t + 1) * (newMean * newMean.transpose())
                                        + newState * newState.transpose()
                                        + eps * maximumVolumeEllipsoid) / t;
            return newCovariance;
        }
    };

    template<typename InternalMatrixType>
    AdaptiveMetropolisProposal<InternalMatrixType>::AdaptiveMetropolisProposal(
            InternalMatrixType A_,
            VectorType b_,
            VectorType currentState_,
            const MatrixType &sqrtMaximumVolumeEllipsoid,
            double eps_,
            unsigned long warmUp_,
            unsigned long t_) :
            A(std::move(A_)),
            b(std::move(b_)),
            state(std::move(currentState_)),
            proposal(this->state),
            t(t_),
            warmUp(warmUp_) {
        if (((b - A * state).array() < boundaryCushion).any()) {
            throw std::invalid_argument("Starting point outside polytope always gives constant Markov chain.");
        }

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

        this->dimensionNames = hops::createDefaultDimensionNames(this->state.rows());
    }

    template<typename InternalMatrixType>
    AdaptiveMetropolisProposal<InternalMatrixType>::AdaptiveMetropolisProposal(
            InternalMatrixType A_,
            VectorType b_,
            VectorType currentState_,
            double eps_,
            unsigned long warmUp_,
            unsigned long t_) :
            A(std::move(A_)),
            b(std::move(b_)),
            state(std::move(currentState_)),
            proposal(this->state),
            t(t_),
            warmUp(warmUp_) {
        if (((b - A * state).array() < boundaryCushion).any()) {
            throw std::invalid_argument("Starting point outside polytope always gives constant Markov chain.");
        }

        normal = std::normal_distribution<double>(0, stepSize);

        // scale down with larger dimensions according to Roberts & Rosenthal, 2001.
        eps = eps_ / A.cols();

        stateMean = state; // actual content is irrelevant as long as dimensions match

        auto MVE = MaximumVolumeEllipsoid<double>::construct(A, b, 10000);
        maximumVolumeEllipsoid = MVE.getEllipsoid();
        stateCovariance = maximumVolumeEllipsoid;
        Eigen::LLT<decltype(maximumVolumeEllipsoid)> solverMaximumVolumeEllipsoid(maximumVolumeEllipsoid);
        choleskyOfMaximumVolumeEllipsoid = MVE.getRoundingTransformation();
        stateCholeskyOfCovariance = MVE.getRoundingTransformation();

        stateLogSqrtDeterminant = stateCholeskyOfCovariance.diagonal().array().log().sum();
        proposalLogSqrtDeterminant = stateLogSqrtDeterminant;
        proposalCovariance = stateCovariance;
    }

    template<typename InternalMatrixType>
    VectorType &AdaptiveMetropolisProposal<InternalMatrixType>::propose(
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

    template<typename InternalMatrixType>
    double AdaptiveMetropolisProposal<InternalMatrixType>::computeLogAcceptanceProbability() {
        bool isProposalInteriorPoint = ((A * proposal - b).array() < -boundaryCushion).all();
        if (!isProposalInteriorPoint) {
            return -std::numeric_limits<double>::infinity();
        }

        proposalCovariance = updateCovariance(stateCovariance, stateMean, proposal);
        Eigen::LLT<decltype(proposalCovariance)> solver(
                proposalCovariance);
        if (solver.info() != Eigen::Success) {
            return -std::numeric_limits<double>::infinity();
        }
        proposalCholeskyOfCovariance = solver.matrixL();

        proposalLogSqrtDeterminant = proposalCholeskyOfCovariance.diagonal().array().log().sum();
        VectorType stateDifference = proposal - state;

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

    template<typename InternalMatrixType>
    VectorType &AdaptiveMetropolisProposal<InternalMatrixType>::acceptProposal() {
        state.swap(proposal);
        stateCovariance = proposalCovariance;
        stateCholeskyOfCovariance = proposalCholeskyOfCovariance;
        stateLogSqrtDeterminant = proposalLogSqrtDeterminant;
        return state;
    }

    template<typename InternalMatrixType>
    void AdaptiveMetropolisProposal<InternalMatrixType>::setState(const VectorType &newState) {
        if (((b - A * newState).array() < boundaryCushion).any()) {
            throw std::invalid_argument("Starting point outside polytope always gives constant Markov chain.");
        }
        AdaptiveMetropolisProposal::state = newState;
    }

    template<typename InternalMatrixType>
    void AdaptiveMetropolisProposal<InternalMatrixType>::setProposal(const VectorType &newProposal) {
        AdaptiveMetropolisProposal::proposal = newProposal;
    }

    template<typename InternalMatrixType>
    std::optional<double> AdaptiveMetropolisProposal<InternalMatrixType>::getStepSize() const {
        return std::nullopt;
    }

    template<typename InternalMatrixType>
    bool AdaptiveMetropolisProposal<InternalMatrixType>::hasStepSize() {
        return false;
    }

    template<typename InternalMatrixType>
    std::string AdaptiveMetropolisProposal<InternalMatrixType>::getProposalName() const {
        return "AdaptiveMetropolis";
    }

    template<typename InternalMatrixType>
    std::unique_ptr<Proposal> AdaptiveMetropolisProposal<InternalMatrixType>::copyProposal() const {
        return std::make_unique<AdaptiveMetropolisProposal>(*this);
    }

    template<typename InternalMatrixType>
    VectorType AdaptiveMetropolisProposal<InternalMatrixType>::getState() const {
        return state;
    }

    template<typename InternalMatrixType>
    VectorType AdaptiveMetropolisProposal<InternalMatrixType>::getProposal() const {
        return proposal;
    }

    template<typename InternalMatrixType>
    void AdaptiveMetropolisProposal<InternalMatrixType>::setParameter(
            const ProposalParameter &parameter, const std::any &value) {
        if (parameter == ProposalParameter::BOUNDARY_CUSHION) {
            this->boundaryCushion = std::any_cast<double>(value);
        } else if (parameter == ProposalParameter::EPSILON) {
            this->eps = std::any_cast<double>(value);
        } else if (parameter == ProposalParameter::WARM_UP) {
            this->warmUp = std::any_cast<long>(value);
        } else {
            throw std::invalid_argument("Can't get parameter which doesn't exist in " + this->getProposalName());
        }
    }

    template<typename InternalMatrixType>
    std::vector<std::string>
    AdaptiveMetropolisProposal<InternalMatrixType>::getParameterNames() const {
        return {
                ProposalParameterName[static_cast<int>(ProposalParameter::BOUNDARY_CUSHION)],
                ProposalParameterName[static_cast<int>(ProposalParameter::EPSILON)],
                ProposalParameterName[static_cast<int>(ProposalParameter::WARM_UP)],
        };
    }

    template<typename InternalMatrixType>
    std::any AdaptiveMetropolisProposal<InternalMatrixType>::getParameter(
            const ProposalParameter &parameter) const {
        if (parameter == ProposalParameter::BOUNDARY_CUSHION) {
            return std::any(boundaryCushion);
        } else if (parameter == ProposalParameter::EPSILON) {
            return std::any(eps);
        } else if (parameter == ProposalParameter::WARM_UP) {
            return std::any(warmUp);
        } else {
            throw std::invalid_argument("Can't get parameter which doesn't exist in " + this->getProposalName());
        }
    }

    template<typename InternalMatrixType>
    std::string AdaptiveMetropolisProposal<InternalMatrixType>::getParameterType(
            const ProposalParameter &parameter) const {
        if (parameter == ProposalParameter::BOUNDARY_CUSHION) {
            return "double";
        } else if (parameter == ProposalParameter::EPSILON) {
            return "double";
        } else if (parameter == ProposalParameter::WARM_UP) {
            return "long";
        } else {
            throw std::invalid_argument("Can't get parameter which doesn't exist in " + this->getProposalName());
        }
    }

    template<typename InternalMatrixType>
    const MatrixType &AdaptiveMetropolisProposal<InternalMatrixType>::getA() const {
        return A;
    }

    template<typename InternalMatrixType>
    const VectorType &AdaptiveMetropolisProposal<InternalMatrixType>::getB() const {
        return b;
    }

    template<typename InternalMatrixType>
    const MatrixType &
    AdaptiveMetropolisProposal<InternalMatrixType>::getCholeskyOfMaximumVolumeEllipsoid() const {
        return choleskyOfMaximumVolumeEllipsoid;
    }

    template<typename InternalMatrixType>
    unsigned long AdaptiveMetropolisProposal<InternalMatrixType>::getT() const {
        return t;
    }

    template<typename InternalMatrixType>
    VectorType &AdaptiveMetropolisProposal<InternalMatrixType>::propose(RandomNumberGenerator &rng,
                                                                        const Eigen::VectorXd &) {
        throw std::runtime_error("Propose with rng and activeIndices not implemented");
    }

    template<typename InternalMatrixType>
    void AdaptiveMetropolisProposal<InternalMatrixType>::setDimensionNames(const std::vector<std::string> &names) {
        dimensionNames = names;
    }

    template<typename InternalMatrixType>
    std::vector<std::string> AdaptiveMetropolisProposal<InternalMatrixType>::getDimensionNames() const {
        return dimensionNames;
    }
}

#endif //HOPS_ADAPTIVEMETROPOLISPROPOSAL_HPP
