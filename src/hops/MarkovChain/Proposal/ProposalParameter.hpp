#ifndef HOPS_PROPOSALPARAMETER_HPP
#define HOPS_PROPOSALPARAMETER_HPP

namespace hops {

    enum class ProposalParameter {
        BOUNDARY_CUSHION,
        EPSILON,
        FISHER_WEIGHT,
        STEP_SIZE,
        WARM_UP,
        MAXIMUM_NUMBER_OF_REFLECTIONS
    };

    static char const* ProposalParameterName[] = {
        "boundary_cushion",
        "epsilon",
        "fisher_weight",
        "step_size",
        "warm_up",
        "maximum_number_of_reflections"
    };
}

#endif //HOPS_PROPOSALPARAMETER_HPP
