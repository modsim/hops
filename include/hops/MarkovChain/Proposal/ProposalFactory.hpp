#ifndef HOPS_PROPOSALFACTORY_HPP
#define HOPS_PROPOSALFACTORY_HPP

#include <hops/MarkovChain/Draw/MetropolisHastingsFilter.hpp>
#include <hops/MarkovChain/ModelMixin.hpp>
#include <hops/MarkovChain/ModelWrapper.hpp>
#include "BallWalk.hpp"
#include "CoordinateHitAndRunProposal.hpp"

namespace hops {
    class ProposalFactory {
    public:
        ProposalFactory() = delete;

        /**
         * @brief Creates uniform proposers.
         */
        template<typename InternalMatrixType, typename InternalVectorType, typename ProposalType>
        static std::unique_ptr<Proposal> createProposal(const InternalMatrixType &inequalityLhs,
                                                        const InternalVectorType &inequalityRhs,
                                                        VectorType startingPoint);
    };

    template <typename InternalMatrixType, typename InternalVectorType, typename ProposalType>
    std::unique_ptr<Proposal>
    ProposalFactory::createProposal(const InternalMatrixType &inequalityLhs, const InternalVectorType &inequalityRhs,
                                    VectorType startingPoint) {
        return std::make_unique<ProposalType>(inequalityLhs, inequalityRhs, startingPoint);
    }
}

#endif //HOPS_PROPOSALFACTORY_HPP
