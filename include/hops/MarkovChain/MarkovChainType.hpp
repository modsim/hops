#ifndef HOPS_MARKOVCHAINTYPE_HPP
#define HOPS_MARKOVCHAINTYPE_HPP

namespace hops {
    enum class MarkovChainType {
        BallWalk,  
        CoordinateHitAndRun,
        CSmMALA,
        CSmMALANoGradient,
        DikinWalk,
        Gaussian,
        HitAndRun,
    };
}

#endif //HOPS_MARKOVCHAINTYPE_HPP
