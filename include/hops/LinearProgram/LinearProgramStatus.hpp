#ifndef HOPS_LINEARPROGRAMSTATUS_HPP
#define HOPS_LINEARPROGRAMSTATUS_HPP

namespace hops {
    enum class LinearProgramStatus {
        UNDEFINED,
        ERROR,
        OPTIMAL,
        INFEASIBLE,
        UNBOUNDED,
    };
}

#endif //HOPS_LINEARPROGRAMSTATUS_HPP
