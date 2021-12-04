#ifndef HOPS_GAUSSIANPROPOSAL_HPP
#define HOPS_GAUSSIANPROPOSAL_HPP

#include <random>

#include <hops/RandomNumberGenerator/RandomNumberGenerator.hpp>
#include <hops/Utility/StringUtility.hpp>

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

        VectorType &propose(RandomNumberGenerator &rng) override;

        VectorType &acceptProposal() override;

        void setState(const VectorType &state) override;

        [[nodiscard]] VectorType getState() const override;

        [[nodiscard]] VectorType getProposal() const override;

        [[nodiscard]] std::vector<std::string> getDimensionNames() const override;

        [[nodiscard]] std::vector<std::string> getParameterNames() const override;

        [[nodiscard]] std::any getParameter(const std::string &parameterName) const override;

        [[nodiscard]] std::string getParameterType(const std::string &name) const override;

        void setParameter(const std::string &parameterName, const std::any &value) override;

        [[nodiscard]] std::optional<double> getStepSize() const;

        void setStepSize(double stepSize);

        [[nodiscard]] std::string getProposalName() const override;

        [[nodiscard]] bool hasStepSize() const override;

        [[nodiscard]] std::unique_ptr<Proposal> deepCopy() const override;

        [[nodiscard]] double computeLogAcceptanceProbability() override;

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
    VectorType &GaussianProposal<InternalMatrixType, InternalVectorType>::propose(RandomNumberGenerator &rng) {
        for (long i = 0; i < proposal.rows(); ++i) {
            proposal(i) = normal(rng);
        }

        proposal.noalias() += state;

        return proposal;
    }

    template<typename InternalMatrixType, typename InternalVectorType>
    VectorType &GaussianProposal<InternalMatrixType, InternalVectorType>::acceptProposal() {
        state.swap(proposal);
        return state;
    }

    template<typename InternalMatrixType, typename InternalVectorType>
    void GaussianProposal<InternalMatrixType, InternalVectorType>::setState(const VectorType &newState) {
        GaussianProposal::state = newState;
    }

    template<typename InternalMatrixType, typename InternalVectorType>
    std::optional<double> GaussianProposal<InternalMatrixType, InternalVectorType>::getStepSize() const {
        return stepSize;
    }

    template<typename InternalMatrixType, typename InternalVectorType>
    void GaussianProposal<InternalMatrixType, InternalVectorType>::setStepSize(double newStepSize) {
        GaussianProposal::stepSize = newStepSize;
        normal = std::normal_distribution<typename InternalMatrixType::Scalar>(0, GaussianProposal::stepSize);
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
    std::vector<std::string> GaussianProposal<InternalMatrixType, InternalVectorType>::getParameterNames() const {
        return {"step_size"};
    }

    template<typename InternalMatrixType, typename InternalVectorType>
    std::any
    GaussianProposal<InternalMatrixType, InternalVectorType>::getParameter(const std::string &parameterName) const {
        std::string lowerCaseParameterName = toLowerCase(parameterName);
        if (lowerCaseParameterName == "step_size") {
            return std::any(stepSize);
        }
        throw std::invalid_argument("Can't get parameter which doesn't exist in " + this->getProposalName());
    }

    template<typename InternalMatrixType, typename InternalVectorType>
    std::string
    GaussianProposal<InternalMatrixType, InternalVectorType>::getParameterType(const std::string &name) const {
        std::string lowerCaseParameterName = toLowerCase(name);
        if (lowerCaseParameterName == "step_size") {
            return "double";
        } else {
            throw std::invalid_argument("Can't get parameter which doesn't exist in " + this->getProposalName());
        }
    }

    template<typename InternalMatrixType, typename InternalVectorType>
    void GaussianProposal<InternalMatrixType, InternalVectorType>::setParameter(const std::string &parameterName,
                                                                                const std::any &value) {
        std::string lowerCaseParameterName = toLowerCase(parameterName);
        if (lowerCaseParameterName == "step_size") {
            setStepSize(std::any_cast<double>(value));
        } else {
            throw std::invalid_argument("Can't get parameter which doesn't exist in " + this->getProposalName());
        }
    }

    template<typename InternalMatrixType, typename InternalVectorType>
    std::vector<std::string> GaussianProposal<InternalMatrixType, InternalVectorType>::getDimensionNames() const {
        std::vector<std::string> names;
        for (long i = 0; i < state.rows(); ++i) {
            names.emplace_back("x_" + std::to_string(i));
        }
        return names;
    }
}

#endif //HOPS_GAUSSIANPROPOSAL_HPP
