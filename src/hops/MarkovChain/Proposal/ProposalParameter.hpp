#ifndef HOPS_PROPOSALPARAMETER_HPP
#define HOPS_PROPOSALPARAMETER_HPP

namespace hops {

    enum class ProposalParameter {
        BOUNDARY_CUSHION,
        EPSILON,
        FISHER_WEIGHT,
        STEP_SIZE,
        WARM_UP
    };

    static char const* ProposalParameterName[] = {
        "boundary_cushion",
        "epsilon",
        "fisher_weight",
        "step_size",
        "warm_up",
    };
    // TODO implement these functions if they are required
//    ProposalParameterName convertString(const std::string& str);
//
//    std::string convertProposalParameterName(const ProposalParameterName& proposalParameterName);

}

#endif //HOPS_PROPOSALPARAMETER_HPP
