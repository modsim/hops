#ifndef HOPS_PROPOSALPARAMETER_HPP
#define HOPS_PROPOSALPARAMETER_HPP

namespace hops {

    enum class ProposalParameter {
        BOUNDARY_CUSHION,
        EPSILON,
        FISHER_WEIGHT,
        STEP_SIZE,
        WARM_UP,
        MAX_REFLECTIONS
    };

    static char const* ProposalParameterName[] = {
        "boundary_cushion",
        "epsilon",
        "fisher_weight",
        "step_size",
        "warm_up",
        "max_reflections"
    };
}

#endif //HOPS_PROPOSALPARAMETER_HPP
