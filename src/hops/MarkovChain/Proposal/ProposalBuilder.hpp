#ifndef HOPS_PROPOSALBUILDER_HPP
#define HOPS_PROPOSALBUILDER_HPP

#include "Proposal.hpp"
#include "hops/Transformation/Transformation.hpp"
#include "hops/Model/Model.hpp"
#include "hops/MarkovChain/Proposal/ParallelTemperingImplementations/ParallelTempering.hpp"

namespace hops {

    class ProposalBuilder {
    public:
        template <typename ProposalImpl, typename ...args>
        std::unique_ptr<Proposal> build();

    private:
        std::unique_ptr<Transformation> transformation = nullptr;
        std::unique_ptr<Model> Model = nullptr;
        std::optional<double> coldness = std::nullopt;
        std::unique_ptr<ParallelTempering> parallelTemperingImpl= nullptr;
        bool isReversibleJump= false;
    };


}

#endif //HOPS_PROPOSALBUILDER_HPP
