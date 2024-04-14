#ifndef HOPS_PROPOSALBUILDER_HPP
#define HOPS_PROPOSALBUILDER_HPP

#include "Proposal.hpp"
#include "hops/Transformation/Transformation.hpp"
#include "hops/Model/Model.hpp"
#include "hops/MarkovChain/Proposal/ParallelTemperingImplementations/ParallelTempering.hpp"

namespace hops {

    class ProposalBuilder {
    public:
        template<typename ProposalImpl, typename ...args>
        std::unique_ptr<Proposal> build() {
            return nullptr;
        }

    private:
        std::unique_ptr<Transformation> transformation = nullptr;
        std::unique_ptr<Model> model = nullptr;
        std::optional<double> coldness = std::nullopt;
        std::unique_ptr<ParallelTempering> parallelTemperingImpl = nullptr;
        std::unique_ptr<Proposal> externalProposal = nullptr;
        bool isReversibleJump = false;
    };


}

#endif //HOPS_PROPOSALBUILDER_HPP
