#ifndef HOPS_MARKOVCHAINTYPE_HPP
#define HOPS_MARKOVCHAINTYPE_HPP

namespace hops {
    enum class MarkovChainType {
        AdaptiveMetropolis,
        BallWalk,
        CoordinateHitAndRun,
        DikinWalk,
        Gaussian,
        HitAndRun,
    };
}

#endif //HOPS_MARKOVCHAINTYPE_HPP
