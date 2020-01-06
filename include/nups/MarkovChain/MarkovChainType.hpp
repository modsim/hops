#ifndef NUPS_MARKOVCHAINTYPE_HPP
#define NUPS_MARKOVCHAINTYPE_HPP

namespace nups {
    enum class MarkovChainType {
        NoOpDraw,
        CoordinateHitAndRun,
        CoordinateHitAndRunRoundedProposals,
        CoordinateHitAndRunRoundedStateSpace,
        DikinWalk,
        HitAndRun,
        HitAndRunRounded,
        HitAndRunRoundedProposals,
    };
}

#endif //NUPS_MARKOVCHAINTYPE_HPP
