#ifndef HOPS_BALLWALKPROPOSAL_HPP
#define HOPS_BALLWALKPROPOSAL_HPP

#include <random>

#include <hops/RandomNumberGenerator/RandomNumberGenerator.hpp>
#include <hops/Utility/StringUtility.hpp>
#include <hops/Utility/VectorType.hpp>

#include "IsSetStepSizeAvailable.hpp"
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

        VectorType &acceptProposal() override;

        void setState(const VectorType &newState) override;

        [[nodiscard]] VectorType getState() const override;

        [[nodiscard]] VectorType getProposal() const override;

        std::optional<std::vector<std::string>> getDimensionNames() const override;

        [[nodiscard]] std::vector<std::string> getParameterNames() const override;

        [[nodiscard]] std::any getParameter(const ProposalParameter &parameter) const override;

        [[nodiscard]] std::string getParameterType(const ProposalParameter &parameter) const override;

        void setParameter(const ProposalParameter &parameter, const std::any &value) override;

        void setStepSize(double stepSize);

        [[nodiscard]] std::string getProposalName() const override;

        [[nodiscard]] std::optional<double> getStepSize() const;

        [[nodiscard]] bool hasStepSize() const override;

        [[nodiscard]] std::unique_ptr<Proposal> copyProposal() const override;

        [[nodiscard]] double computeLogAcceptanceProbability() override;

    private:
        InternalMatrixType A;
        InternalVectorType b;
        VectorType state;
        VectorType proposal;

        double stepSize;

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
            stepSize(stepSize_) {}

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
        return "BallWalkProposal";
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
    bool BallWalkProposal<InternalMatrixType, InternalVectorType>::hasStepSize() const {
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
    BallWalkProposal<InternalMatrixType, InternalVectorType>::getParameterType(const ProposalParameter &parameter) const {
        if (parameter == ProposalParameter::STEP_SIZE) {
            return "double";
        } else {
            throw std::invalid_argument("Can't get parameter which doesn't exist in " + this->getProposalName());
        }
    }

    template<typename InternalMatrixType, typename InternalVectorType>
    std::optional<std::vector<std::string>> BallWalkProposal<InternalMatrixType, InternalVectorType>::getDimensionNames() const {
        std::vector<std::string> names;
        for (long i = 0; i < state.rows(); ++i) {
            names.emplace_back("x_" + std::to_string(i));
        }
        return names;
    }
}


#endif //HOPS_BALLWALKPROPOSAL_HPP
