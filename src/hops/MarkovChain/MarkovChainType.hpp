#ifndef HOPS_MARKOVCHAINTYPE_HPP
#define HOPS_MARKOVCHAINTYPE_HPP

#include <string>

namespace hops {
    enum class MarkovChainType {
        AdaptiveMetropolis,
        BallWalk,
        BilliardAdaptiveMetropolis,
        BilliardMALA,
        CoordinateHitAndRun,
        CSmMALA,
        DikinWalk,
        Gaussian,
        HitAndRun,
        TruncatedGaussian,
    };

    std::string markovChainTypeToFullString(MarkovChainType markovChainType);

    std::string markovChainTypeToShortString(MarkovChainType markovChainType);

    bool checkIfStringIsEqualToChainType(const std::string &chainTypeString, MarkovChainType markovChainType);

    MarkovChainType stringToMarkovChainType(const std::string &chainString);
}

#endif //HOPS_MARKOVCHAINTYPE_HPP
