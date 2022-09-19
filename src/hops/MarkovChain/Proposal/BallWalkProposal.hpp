#ifndef HOPS_BALLWALKPROPOSAL_HPP
#define HOPS_BALLWALKPROPOSAL_HPP

#include <optional>
#include <random>

#include "hops/RandomNumberGenerator/RandomNumberGenerator.hpp"
#include "hops/Utility/MatrixType.hpp"
#include "hops/Utility/StringUtility.hpp"
#include "hops/Utility/VectorType.hpp"

#include "Proposal.hpp"

namespace hops {
    template<typename InternalMatrixType, typename InternalVectorType>
    class BallWalkProposal : public Proposal {
    public:
        /**
         * @brief Constructs BallWalk proposal mechanism on polytope defined as Ax<b.
         * @param A
         * @param b
         * @param currentState
         * @param stepSize The radius of the ball from which the proposal move is drawn
         */
        BallWalkProposal(InternalMatrixType A, InternalVectorType b, VectorType currentState, double stepSize = 1);

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

        void setStepSize(double stepSize);

        [[nodiscard]] std::string getProposalName() const override;

        [[nodiscard]] std::optional<double> getStepSize() const override;

        [[nodiscard]] static bool hasStepSize();

        [[nodiscard]] std::unique_ptr<Proposal> copyProposal() const override;

        [[nodiscard]] double computeLogAcceptanceProbability() override;

        [[nodiscard]] const MatrixType &getA() const override;

        [[nodiscard]] const VectorType &getB() const override;

    private:
        InternalMatrixType A;
        InternalVectorType b;
        VectorType state;
        VectorType proposal;

        double stepSize;

        std::vector<std::string> dimensionNames;

        std::uniform_real_distribution<typename InternalMatrixType::Scalar> uniform;
        std::normal_distribution<typename InternalMatrixType::Scalar> normal;
    };

    template<typename InternalMatrixType, typename InternalVectorType>
    BallWalkProposal<InternalMatrixType, InternalVectorType>::BallWalkProposal(InternalMatrixType A_,
                                                                               InternalVectorType b_,
                                                                               VectorType currentState_,
                                                                               double stepSize_) :
            A(std::move(A_)),
            b(std::move(b_)),
            state(std::move(currentState_)),
            proposal(this->state),
            stepSize(stepSize_) {
        if (((b - A * state).array() < 0).any()) {
            throw std::invalid_argument("Starting point outside polytope always gives constant Markov chain.");
        }
        this->dimensionNames = hops::createDefaultDimensionNames(this->state.rows());
    }


    template<typename InternalMatrixType, typename InternalVectorType>
    VectorType &BallWalkProposal<InternalMatrixType, InternalVectorType>::propose(
            RandomNumberGenerator &randomNumberGenerator) {
        // Creates proposal on Ballsurface
        for (long i = 0; i < proposal.rows(); ++i) {
            proposal(i) = normal(randomNumberGenerator);
        }
        proposal.normalize();
        // Scales proposal to radius of Ball
        proposal.noalias() = stepSize * proposal;
        // Scales proposal into Ball
        proposal.noalias() = std::pow(uniform(randomNumberGenerator), 1. / proposal.rows()) * proposal;
        proposal.noalias() += state;

        return proposal;
    }

    template<typename InternalMatrixType, typename InternalVectorType>
    VectorType &BallWalkProposal<InternalMatrixType, InternalVectorType>::acceptProposal() {
        state.swap(proposal);
        return state;
    }

    template<typename InternalMatrixType, typename InternalVectorType>
    void BallWalkProposal<InternalMatrixType, InternalVectorType>::setState(const VectorType &newState) {
        BallWalkProposal::state = newState;
        if (((b - A * state).array() < 0).any()) {
            throw std::invalid_argument("Starting point outside polytope always gives constant Markov chain.");
        }
    }

    template<typename InternalMatrixType, typename InternalVectorType>
    void BallWalkProposal<InternalMatrixType, InternalVectorType>::setProposal(const VectorType &newProposal) {
        BallWalkProposal::proposal = newProposal;
    }

    template<typename InternalMatrixType, typename InternalVectorType>
    void BallWalkProposal<InternalMatrixType, InternalVectorType>::setStepSize(double newStepSize) {
        BallWalkProposal::stepSize = newStepSize;
    }

    template<typename InternalMatrixType, typename InternalVectorType>
    std::optional<double> BallWalkProposal<InternalMatrixType, InternalVectorType>::getStepSize() const {
        return stepSize;
    }

    template<typename InternalMatrixType, typename InternalVectorType>
    std::string BallWalkProposal<InternalMatrixType, InternalVectorType>::getProposalName() const {
        return "BallWalk";
    }

    template<typename InternalMatrixType, typename InternalVectorType>
    double BallWalkProposal<InternalMatrixType, InternalVectorType>::computeLogAcceptanceProbability() {
        bool isProposalInteriorPoint = ((A * proposal - b).array() < 0).all();
        if (!isProposalInteriorPoint) {
            return -std::numeric_limits<typename InternalMatrixType::Scalar>::infinity();
        }
        return 0;
    }

    template<typename InternalMatrixType, typename InternalVectorType>
    bool BallWalkProposal<InternalMatrixType, InternalVectorType>::hasStepSize() {
        return true;
    }

    template<typename InternalMatrixType, typename InternalVectorType>
    std::unique_ptr<Proposal> BallWalkProposal<InternalMatrixType, InternalVectorType>::copyProposal() const {
        return std::make_unique<BallWalkProposal>(*this);
    }

    template<typename InternalMatrixType, typename InternalVectorType>
    VectorType BallWalkProposal<InternalMatrixType, InternalVectorType>::getState() const {
        return state;
    }

    template<typename InternalMatrixType, typename InternalVectorType>
    VectorType BallWalkProposal<InternalMatrixType, InternalVectorType>::getProposal() const {
        return proposal;
    }

    template<typename InternalMatrixType, typename InternalVectorType>
    void BallWalkProposal<InternalMatrixType, InternalVectorType>::setParameter(const ProposalParameter &parameter,
                                                                                const std::any &value) {
        if (parameter == ProposalParameter::STEP_SIZE) {
            setStepSize(std::any_cast<double>(value));
        } else {
            throw std::invalid_argument("Can't get parameter which doesn't exist in " + this->getProposalName());
        }
    }

    template<typename InternalMatrixType, typename InternalVectorType>
    std::vector<std::string> BallWalkProposal<InternalMatrixType, InternalVectorType>::getParameterNames() const {
        return {ProposalParameterName[static_cast<int>(ProposalParameter::STEP_SIZE)]};
    }

    template<typename InternalMatrixType, typename InternalVectorType>
    std::any
    BallWalkProposal<InternalMatrixType, InternalVectorType>::getParameter(const ProposalParameter &parameter) const {
        if (parameter == ProposalParameter::STEP_SIZE) {
            return std::any(stepSize);
        } else {
            throw std::invalid_argument("Can't get parameter which doesn't exist in " + this->getProposalName());
        }
    }

    template<typename InternalMatrixType, typename InternalVectorType>
    std::string
    BallWalkProposal<InternalMatrixType, InternalVectorType>::getParameterType(
            const ProposalParameter &parameter) const {
        if (parameter == ProposalParameter::STEP_SIZE) {
            return "double";
        } else {
            throw std::invalid_argument("Can't get parameter which doesn't exist in " + this->getProposalName());
        }
    }

    template<typename InternalMatrixType, typename InternalVectorType>
    const MatrixType &BallWalkProposal<InternalMatrixType, InternalVectorType>::getA() const {
        return A;
    }

    template<typename InternalMatrixType, typename InternalVectorType>
    const VectorType &BallWalkProposal<InternalMatrixType, InternalVectorType>::getB() const {
        return b;
    }

    template<typename InternalMatrixType, typename InternalVectorType>
    VectorType &BallWalkProposal<InternalMatrixType, InternalVectorType>::propose(RandomNumberGenerator &rng,
                                                                                  const Eigen::VectorXd &activeIndices) {
        throw std::runtime_error("Propose with rng and activeIndices not implemented");
    }

    template<typename InternalMatrixType, typename InternalVectorType>
    void
    BallWalkProposal<InternalMatrixType, InternalVectorType>::setDimensionNames(const std::vector<std::string> &names) {
        dimensionNames = names;
    }

    template<typename InternalMatrixType, typename InternalVectorType>
    std::vector<std::string> BallWalkProposal<InternalMatrixType, InternalVectorType>::getDimensionNames() const {
        return dimensionNames;
    }
}


#endif //HOPS_BALLWALKPROPOSAL_HPP
