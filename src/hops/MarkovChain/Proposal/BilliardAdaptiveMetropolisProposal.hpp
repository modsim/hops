#ifndef HOPS_BILLIARDADAPTIVEMETROPOLISPROPOSAL_HPP
#define HOPS_BILLIARDADAPTIVEMETROPOLISPROPOSAL_HPP

#include "AdaptiveMetropolisProposal.hpp"
#include "Reflector.hpp"

namespace hops {
    template<typename InternalMatrixType = MatrixType>
    class BilliardAdaptiveMetropolisProposal
            : public AdaptiveMetropolisProposal<InternalMatrixType> {
    public:
        BilliardAdaptiveMetropolisProposal(
                const AdaptiveMetropolisProposal<InternalMatrixType> &adaptiveMetropolisProposal,
                long maxReflections);

        BilliardAdaptiveMetropolisProposal(InternalMatrixType A,
                                           VectorType b,
                                           const VectorType &currentState,
                                           double eps = 1.e-3,
                                           unsigned long warmUp = 100,
                                           unsigned long t = 0,
                                           long maxReflections = 100);


        BilliardAdaptiveMetropolisProposal(InternalMatrixType A,
                                           VectorType b,
                                           const VectorType &currentState,
                                           const MatrixType &sqrtMaximumVolumeEllipsoid,
                                           double eps = 1.e-3,
                                           unsigned long warmUp = 100,
                                           unsigned long t = 0,
                                           long maxReflections = 100);


        VectorType &propose(RandomNumberGenerator &randomNumberGenerator) override;

        VectorType &propose(RandomNumberGenerator &rng, const Eigen::VectorXd &activeIndices) override;

        void setDimensionNames(const std::vector<std::string> &names) override;

        [[nodiscard]] std::vector<std::string> getDimensionNames() const override;

        [[nodiscard]] std::vector<std::string> getParameterNames() const override;

        [[nodiscard]] std::any getParameter(const ProposalParameter &parameter) const override;

        [[nodiscard]] std::string getParameterType(const ProposalParameter &parameter) const override;

        void setParameter(const ProposalParameter &parameter, const std::any &value) override;

        [[nodiscard]] std::string getProposalName() const override;

        [[nodiscard]] std::unique_ptr<Proposal> copyProposal() const override;

    private:
        long maxReflections;
    };

    template<typename InternalMatrixType>
    BilliardAdaptiveMetropolisProposal<InternalMatrixType>::BilliardAdaptiveMetropolisProposal(
            const AdaptiveMetropolisProposal<InternalMatrixType> &adaptiveMetropolisProposal,
            long maxReflections) :
            AdaptiveMetropolisProposal<InternalMatrixType>(adaptiveMetropolisProposal),
            maxReflections(maxReflections) {}

    template<typename InternalMatrixType>
    BilliardAdaptiveMetropolisProposal<InternalMatrixType>::BilliardAdaptiveMetropolisProposal(
            InternalMatrixType A,
            VectorType b,
            const VectorType &currentState,
            double eps,
            unsigned long warmUp,
            unsigned long t,
            long maxReflections): AdaptiveMetropolisProposal<InternalMatrixType>(std::move(A),
                                                                                 std::move(b),
                                                                                 currentState,
                                                                                 eps,
                                                                                 warmUp,
                                                                                 t),
                                  maxReflections(maxReflections) {}

    template<typename InternalMatrixType>
    BilliardAdaptiveMetropolisProposal<InternalMatrixType>::BilliardAdaptiveMetropolisProposal(InternalMatrixType A,
                                                                                               VectorType b,
                                                                                               const VectorType &currentState,
                                                                                               const MatrixType &sqrtMaximumVolumeEllipsoid,
                                                                                               double eps,
                                                                                               unsigned long warmUp,
                                                                                               unsigned long t,
                                                                                               long maxReflections)
            : AdaptiveMetropolisProposal<InternalMatrixType>(std::move(A),
                                                             std::move(b),
                                                             currentState,
                                                             sqrtMaximumVolumeEllipsoid,
                                                             eps,
                                                             warmUp,
                                                             t),
              maxReflections(maxReflections) {}

    template<typename InternalMatrixType>
    VectorType &BilliardAdaptiveMetropolisProposal<InternalMatrixType>::propose(
            RandomNumberGenerator &randomNumberGenerator) {
        VectorType &proposal = AdaptiveMetropolisProposal<InternalMatrixType>::propose(
                randomNumberGenerator);
        const VectorType &state = AdaptiveMetropolisProposal<InternalMatrixType>::getState();


        const auto &reflectionResult = Reflector::reflectIntoPolytope(this->A, this->b, state, proposal,
                                                                      maxReflections);
        proposal = std::get<2>(reflectionResult);
        return proposal;
    }

    template<typename InternalMatrixType>
    std::vector<std::string>
    BilliardAdaptiveMetropolisProposal<InternalMatrixType>::getParameterNames() const {
        std::vector<std::string> parameterNames = AdaptiveMetropolisProposal<InternalMatrixType>::getParameterNames();
        parameterNames.template emplace_back("max_reflections");
        return parameterNames;
    }

    template<typename InternalMatrixType>
    std::any BilliardAdaptiveMetropolisProposal<InternalMatrixType>::getParameter(
            const ProposalParameter &parameter) const {
        if (parameter == ProposalParameter::MAX_REFLECTIONS) {
            return std::any(this->maxReflections);
        }
        return AdaptiveMetropolisProposal<InternalMatrixType>::getParameter(parameter);
    }

    template<typename InternalMatrixType>
    std::string BilliardAdaptiveMetropolisProposal<InternalMatrixType>::getParameterType(
            const ProposalParameter &parameter) const {
        if (parameter == ProposalParameter::MAX_REFLECTIONS) {
            return "long";
        }
        return AdaptiveMetropolisProposal<InternalMatrixType>::getParameterType(parameter);
    }

    template<typename InternalMatrixType>
    void
    BilliardAdaptiveMetropolisProposal<InternalMatrixType>::setParameter(
            const ProposalParameter &parameter,
            const std::any &value) {
        if (parameter == ProposalParameter::MAX_REFLECTIONS) {
            maxReflections = std::any_cast<long>(value);
        } else {
            AdaptiveMetropolisProposal<InternalMatrixType>::setParameter(parameter, value);
        }
    }

    template<typename InternalMatrixType>
    std::string BilliardAdaptiveMetropolisProposal<InternalMatrixType>::getProposalName() const {
        return "BilliardAdaptiveMetropolis";
    }

    template<typename InternalMatrixType>
    std::unique_ptr<Proposal>
    BilliardAdaptiveMetropolisProposal<InternalMatrixType>::copyProposal() const {
        return std::make_unique<BilliardAdaptiveMetropolisProposal>(*this);
    }

    template<typename InternalMatrixType>
    VectorType &BilliardAdaptiveMetropolisProposal<InternalMatrixType>::propose(RandomNumberGenerator &rng,
                                                                                const Eigen::VectorXd &activeIndices) {
        throw std::runtime_error("Propose with rng and activeIndices not implemented");
    }

    template<typename InternalMatrixType>
    void
    BilliardAdaptiveMetropolisProposal<InternalMatrixType>::setDimensionNames(const std::vector<std::string> &names) {
        AdaptiveMetropolisProposal<InternalMatrixType>::setDimensionNames(names);
    }

    template<typename InternalMatrixType>
    std::vector<std::string> BilliardAdaptiveMetropolisProposal<InternalMatrixType>::getDimensionNames() const {
        return AdaptiveMetropolisProposal<InternalMatrixType>::getDimensionNames();
    }
}

#endif //HOPS_BILLIARDADAPTIVEMETROPOLISPROPOSAL_HPP
