#ifndef HOPS_PROPOSALPARAMETER_HPP
#define HOPS_PROPOSALPARAMETER_HPP

namespace hops {

    enum class ProposalParameter {
        BOUNDARY_CUSHION,
        COLDNESS,
        EPSILON,
        FISHER_WEIGHT,
        STEP_SIZE,
        WARM_UP,
        MAX_REFLECTIONS,
        MODEL_JUMP_PROBABILITY,
        ACTIVATION_PROBABILITY,
        DEACTIVATION_PROBABILITY,
    };

    __attribute__((unused)) static char const *ProposalParameterName[] = {
            "boundary_cushion",
            "coldness",
            "epsilon",
            "fisher_weight",
            "step_size",
            "warm_up",
            "max_reflections",
            "model_jump_probability",
            "activation_probability",
            "deactivation_probability"
    };
}

#endif //HOPS_PROPOSALPARAMETER_HPP
