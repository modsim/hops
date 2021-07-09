#ifndef HOPS_GAUSSIANPROPOSAL_HPP
#define HOPS_GAUSSIANPROPOSAL_HPP

#include "../IsSetStepSizeAvailable.hpp"
#include "../../RandomNumberGenerator/RandomNumberGenerator.hpp"
#include "../../FileWriter/CsvWriter.hpp"
#include <random>

namespace hops {
    template<typename MatrixType, typename VectorType>
    class GaussianProposal {
    public:
        using StateType = VectorType;

        /**
         * @brief Constructs classical Gaussian random walk proposal mechanism on polytope defined as Ax<b.
         * @param A
         * @param b
         * @param currentState
         * @param stepSize         The standard deviation of the isotropic Gaussian proposal distribution
         */
        GaussianProposal(MatrixType A, VectorType b, StateType currentState, typename MatrixType::Scalar stepSize = 1);

        void propose(RandomNumberGenerator &randomNumberGenerator);

        void acceptProposal();

        StateType getState() const;

        StateType getProposal() const;

        void setState(StateType newState);

        void setStepSize(typename MatrixType::Scalar stepSize);

        typename MatrixType::Scalar getStepSize() const;

        [[nodiscard]] typename MatrixType::Scalar computeLogAcceptanceProbability() {
            bool isProposalInteriorPoint = ((b - A * proposal).array() >= 0).all();
            if (!isProposalInteriorPoint) {
                return -std::numeric_limits<typename MatrixType::Scalar>::infinity();
            }
            return 0;
        }

        std::string getName();

    private:
        MatrixType A;
        VectorType b;
        StateType state;
        StateType proposal;

        typename MatrixType::Scalar stepSize;

        std::normal_distribution<typename MatrixType::Scalar> normal;
    };

    template<typename MatrixType, typename VectorType>
    GaussianProposal<MatrixType, VectorType>::GaussianProposal(MatrixType A_,
                                                               VectorType b_,
                                                               VectorType currentState_,
                                                               typename MatrixType::Scalar stepSize_) :
            A(std::move(A_)),
            b(std::move(b_)),
            state(std::move(currentState_)),
            proposal(this->state),
            stepSize(stepSize_) {
        normal = std::normal_distribution<typename MatrixType::Scalar>(0, stepSize);
    }

    template<typename MatrixType, typename VectorType>
    void GaussianProposal<MatrixType, VectorType>::propose(
            RandomNumberGenerator &randomNumberGenerator) {
        for (long i = 0; i < proposal.rows(); ++i) {
            proposal(i) = normal(randomNumberGenerator);
        }

        proposal.noalias() += state;
    }

    template<typename MatrixType, typename VectorType>
    void
    GaussianProposal<MatrixType, VectorType>::acceptProposal() {
        state.swap(proposal);
    }

    template<typename MatrixType, typename VectorType>
    typename GaussianProposal<MatrixType, VectorType>::StateType
    GaussianProposal<MatrixType, VectorType>::getState() const {
        return state;
    }

    template<typename MatrixType, typename VectorType>
    typename GaussianProposal<MatrixType, VectorType>::StateType
    GaussianProposal<MatrixType, VectorType>::getProposal() const {
        return proposal;
    }

    template<typename MatrixType, typename VectorType>
    void GaussianProposal<MatrixType, VectorType>::setState(VectorType newState) {
        GaussianProposal::state = std::move(newState);
    }

    template<typename MatrixType, typename VectorType>
    void GaussianProposal<MatrixType, VectorType>::setStepSize(
            typename MatrixType::Scalar newStepSize) {
        stepSize = newStepSize;
        normal = std::normal_distribution<typename MatrixType::Scalar>(0, stepSize);
    }

    template<typename MatrixType, typename VectorType>
    typename MatrixType::Scalar
    GaussianProposal<MatrixType, VectorType>::getStepSize() const {
        return stepSize;
    }

    template<typename MatrixType, typename VectorType>
    std::string GaussianProposal<MatrixType, VectorType>::getName() {
        return "Gaussian";
    }
}

#endif //HOPS_GAUSSIANPROPOSAL_HPP
