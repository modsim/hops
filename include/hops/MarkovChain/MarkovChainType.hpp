#ifndef HOPS_MARKOVCHAINTYPE_HPP
#define HOPS_MARKOVCHAINTYPE_HPP

namespace hops {
    enum class MarkovChainType {
        NoOpDraw,
        CoordinateHitAndRun,
        CoordinateHitAndRunRoundedStateSpace,
        DikinWalk,
        HitAndRun,
        HitAndRunRounded,
        HitAndRunRoundedProposals,
    };
}

#endif //HOPS_MARKOVCHAINTYPE_HPP
