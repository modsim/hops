#ifndef HOPS_BILLIARDADAPTIVEMETROPOLISPROPOSAL_HPP
#define HOPS_BILLIARDADAPTIVEMETROPOLISPROPOSAL_HPP

#include "AdaptiveMetropolisProposal.hpp"
#include "Reflector.hpp"

namespace hops {
    template<typename InternalMatrixType = MatrixType, typename InternalVectorType = VectorType>
    class BilliardAdaptiveMetropolisProposal
            : public AdaptiveMetropolisProposal<InternalMatrixType, InternalVectorType> {
    public:
        using StateType = VectorType;

        BilliardAdaptiveMetropolisProposal(
                const AdaptiveMetropolisProposal<InternalMatrixType, InternalVectorType> &adaptiveMetropolisProposal,
                long maximumNumberOfReflections);

        VectorType &propose(RandomNumberGenerator &randomNumberGenerator) override;

        [[nodiscard]] std::vector<std::string> getParameterNames() const override;

        [[nodiscard]] std::any getParameter(const ProposalParameter &parameter) const override;

        [[nodiscard]] std::string getParameterType(const ProposalParameter &parameter) const override;

        void setParameter(const ProposalParameter &parameter, const std::any &value) override;

        [[nodiscard]] std::string getProposalName() const override;

        [[nodiscard]] std::unique_ptr<Proposal> copyProposal() const override;

    private:
        long maxNumberOfReflections;
    };

    template<typename InternalMatrixType, typename InternalVectorType>
    BilliardAdaptiveMetropolisProposal<InternalMatrixType, InternalVectorType>::BilliardAdaptiveMetropolisProposal(
            const AdaptiveMetropolisProposal<InternalMatrixType, InternalVectorType> &adaptiveMetropolisProposal,
            long maximumNumberOfReflections) :
            AdaptiveMetropolisProposal<InternalMatrixType, InternalVectorType>(adaptiveMetropolisProposal),
            maxNumberOfReflections(maximumNumberOfReflections) {}

    template<typename InternalMatrixType, typename InternalVectorType>
    VectorType &BilliardAdaptiveMetropolisProposal<InternalMatrixType, InternalVectorType>::propose(
            RandomNumberGenerator &randomNumberGenerator) {
        VectorType &proposal = AdaptiveMetropolisProposal<InternalMatrixType, InternalVectorType>::propose(
                randomNumberGenerator);
        const VectorType &state = AdaptiveMetropolisProposal<InternalMatrixType, InternalVectorType>::getState();


        const auto &reflectionResult = Reflector::reflectIntoPolytope(this->A, this->b, state, proposal, maxNumberOfReflections);
        if (this->isTrackingOfProposalStatisticsActivated()) {
            ProposalStatistics &infos = this->getProposalStatistics();
            infos.appendInfo("reflection_successful", std::get<0>(reflectionResult));
            infos.appendInfo("number_of_reflections", std::get<1>(reflectionResult));
        }

        proposal = std::get<2>(reflectionResult);
        return proposal;
    }

    template<typename InternalMatrixType, typename InternalVectorType>
    std::vector<std::string>
    BilliardAdaptiveMetropolisProposal<InternalMatrixType, InternalVectorType>::getParameterNames() const {
        std::vector<std::string> parameterNames = AdaptiveMetropolisProposal<InternalMatrixType, InternalVectorType>::getParameterNames();
        parameterNames.template emplace_back("maximum_number_of_reflections");
        return parameterNames;
    }

    template<typename InternalMatrixType, typename InternalVectorType>
    std::any BilliardAdaptiveMetropolisProposal<InternalMatrixType, InternalVectorType>::getParameter(
            const ProposalParameter &parameter) const {
        if (parameter == ProposalParameter::MAXIMUM_NUMBER_OF_REFLECTIONS) {
            return std::any(this->maxNumberOfReflections);
        }
        return AdaptiveMetropolisProposal<InternalMatrixType, InternalVectorType>::getParameter(parameter);
    }

    template<typename InternalMatrixType, typename InternalVectorType>
    std::string BilliardAdaptiveMetropolisProposal<InternalMatrixType, InternalVectorType>::getParameterType(
            const ProposalParameter &parameter) const {
        if (parameter == ProposalParameter::MAXIMUM_NUMBER_OF_REFLECTIONS) {
            return "long";
        }
        return AdaptiveMetropolisProposal<InternalMatrixType, InternalVectorType>::getParameterType(parameter);
    }

    template<typename InternalMatrixType, typename InternalVectorType>
    void
    BilliardAdaptiveMetropolisProposal<InternalMatrixType, InternalVectorType>::setParameter(
            const ProposalParameter &parameter,
            const std::any &value) {
        if (parameter == ProposalParameter::MAXIMUM_NUMBER_OF_REFLECTIONS) {
            maxNumberOfReflections = std::any_cast<long>(value);
        } else {
            AdaptiveMetropolisProposal<InternalMatrixType, InternalVectorType>::setParameter(parameter, value);
        }
    }

    template<typename InternalMatrixType, typename InternalVectorType>
    std::string BilliardAdaptiveMetropolisProposal<InternalMatrixType, InternalVectorType>::getProposalName() const {
        return "BilliardAdaptiveMetropolis";
    }

    template<typename InternalMatrixType, typename InternalVectorType>
    std::unique_ptr<Proposal>
    BilliardAdaptiveMetropolisProposal<InternalMatrixType, InternalVectorType>::copyProposal() const {
        return std::make_unique<BilliardAdaptiveMetropolisProposal>(*this);
    }
}

#endif //HOPS_BILLIARDADAPTIVEMETROPOLISPROPOSAL_HPP
