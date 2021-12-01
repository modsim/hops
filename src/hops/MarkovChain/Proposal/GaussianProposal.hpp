#ifndef HOPS_GAUSSIANPROPOSAL_HPP
#define HOPS_GAUSSIANPROPOSAL_HPP

#include <random>
#include <hops/RandomNumberGenerator/RandomNumberGenerator.hpp>

#include "IsSetStepSizeAvailable.hpp"
#include "Proposal.hpp"

namespace hops {
    template<typename InternalMatrixType, typename InternalVectorType>
    class GaussianProposal : public Proposal {
    public:

        /**
         * @brief Constructs classical Gaussian random walk proposal mechanism on polytope defined as Ax<b.
         * @param A
         * @param b
         * @param currentState
         * @param stepSize         The standard deviation of the isotropic Gaussian proposal distribution
         */
        GaussianProposal(InternalMatrixType A, InternalVectorType b, VectorType currentState, double stepSize = 1);

        std::pair<double, VectorType> propose(RandomNumberGenerator &rng) override;

        VectorType acceptProposal() override;

        void setState(VectorType state) override;

        [[nodiscard]] VectorType getState() const override;

        [[nodiscard]] VectorType getProposal() const override;

        void setParameter(ProposalParameterName parameterName, const std::any &value) override;

        [[nodiscard]] std::optional<double> getStepSize() const;

        void setStepSize(double stepSize);

        [[nodiscard]] std::string getProposalName() const override;

        [[nodiscard]] bool hasStepSize() const override;

        [[nodiscard]] std::unique_ptr<Proposal> deepCopy() const override;

        [[nodiscard]] double computeLogAcceptanceProbability();

    private:
        InternalMatrixType A;
        InternalVectorType b;
        VectorType state;
        VectorType proposal;

        double stepSize;

        std::normal_distribution<typename InternalMatrixType::Scalar> normal;
    };

    template<typename InternalMatrixType, typename InternalVectorType>
    GaussianProposal<InternalMatrixType, InternalVectorType>::GaussianProposal(InternalMatrixType A_,
                                                               InternalVectorType b_,
                                                               VectorType currentState_,
                                                               double stepSize_) :
            A(std::move(A_)),
            b(std::move(b_)),
            state(std::move(currentState_)),
            proposal(this->state),
            stepSize(stepSize_) {
        normal = std::normal_distribution<typename InternalMatrixType::Scalar>(0, stepSize);
    }

    template<typename InternalMatrixType, typename InternalVectorType>
    std::pair<double, VectorType> GaussianProposal<InternalMatrixType, InternalVectorType>::propose(RandomNumberGenerator &rng) {
        for (long i = 0; i < proposal.rows(); ++i) {
            proposal(i) = normal(rng);
        }

        proposal.noalias() += state;

        return {computeLogAcceptanceProbability(), proposal};
    }

    template<typename InternalMatrixType, typename InternalVectorType>
    VectorType GaussianProposal<InternalMatrixType, InternalVectorType>::acceptProposal() {
        state.swap(proposal);
        return state;
    }

    template<typename InternalMatrixType, typename InternalVectorType>
    void GaussianProposal<InternalMatrixType, InternalVectorType>::setState(VectorType state) {
        GaussianProposal::state = std::move(state);
    }

    template<typename InternalMatrixType, typename InternalVectorType>
    std::optional<double> GaussianProposal<InternalMatrixType, InternalVectorType>::getStepSize() const {
        return stepSize;
    }

    template<typename InternalMatrixType, typename InternalVectorType>
    void GaussianProposal<InternalMatrixType, InternalVectorType>::setStepSize(double stepSize) {
        GaussianProposal::stepSize = stepSize;
        normal = std::normal_distribution<typename InternalMatrixType::Scalar>(0, stepSize);
    }

    template<typename InternalMatrixType, typename InternalVectorType>
    std::string GaussianProposal<InternalMatrixType, InternalVectorType>::getProposalName() const {
        return "Gaussian";
    }

    template<typename InternalMatrixType, typename InternalVectorType>
    double GaussianProposal<InternalMatrixType, InternalVectorType>::computeLogAcceptanceProbability() {
        bool isProposalInteriorPoint = ((b - A * proposal).array() >= 0).all();
        if (!isProposalInteriorPoint) {
            return -std::numeric_limits<double>::infinity();
        }
        return 0;
    }

    template<typename InternalMatrixType, typename InternalVectorType>
    bool GaussianProposal<InternalMatrixType, InternalVectorType>::hasStepSize() const {
        return true;
    }

    template<typename InternalMatrixType, typename InternalVectorType>
    std::unique_ptr<Proposal> GaussianProposal<InternalMatrixType, InternalVectorType>::deepCopy() const {
        return std::make_unique<GaussianProposal>(*this);
    }

    template<typename InternalMatrixType, typename InternalVectorType>
    VectorType GaussianProposal<InternalMatrixType, InternalVectorType>::getState() const {
        return state;
    }

    template<typename InternalMatrixType, typename InternalVectorType>
    VectorType GaussianProposal<InternalMatrixType, InternalVectorType>::getProposal() const {
        return proposal;
    }

    template<typename InternalMatrixType, typename InternalVectorType>
    void GaussianProposal<InternalMatrixType, InternalVectorType>::setParameter(ProposalParameterName parameterName,
                                                                                const std::any &value) {
        switch (parameterName) {
            case ProposalParameterName::STEP_SIZE: {
                setStepSize(std::any_cast<double>(value));
                break;
            }
            default:
                throw std::invalid_argument("Can't set parameter which doesn't exist in GaussianProposal.");
        }

    }
}

#endif //HOPS_GAUSSIANPROPOSAL_HPP
