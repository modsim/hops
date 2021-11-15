#ifndef HOPS_MARKOVCHAINTYPE_HPP
#define HOPS_MARKOVCHAINTYPE_HPP

namespace hops {
    enum class MarkovChainType {
        AdaptiveMetropolis,
        BallWalk,
        CoordinateHitAndRun,
        CSmMALA,
        DikinWalk,
        Gaussian,
        HitAndRun,
    };


    std::string MarkovChainTypeToFullString(MarkovChainType markovChainType) {
        switch(markovChainType) {
            case MarkovChainType::AdaptiveMetropolis:
                return "Adaptive Metropolis";
            case MarkovChainType::BallWalk:
                return "Ball Walk";
            case MarkovChainType::CoordinateHitAndRun:
                return "Coordinate Hit-and-Run";
            case MarkovChainType::CSmMALA:
                return "Constrained Simplified manifold Metropolis Adjusted Langevin Algorithm";
            case MarkovChainType::DikinWalk:
                return "Dikin Walk";
            case MarkovChainType::Gaussian:
                return "Gaussian Random Walk";
            case MarkovChainType::HitAndRun:
                return "Hit-and-Run";
            default:
                throw std::runtime_error("Bug in switch case.");
        }
    }

    std::string MarkovChainTypeToShortcutString(MarkovChainType markovChainType) {
        switch(markovChainType) {
            case MarkovChainType::AdaptiveMetropolis:
                return "AM";
            case MarkovChainType::BallWalk:
                return "BW";
            case MarkovChainType::CoordinateHitAndRun:
                return "CHR";
            case MarkovChainType::CSmMALA:
                return "CSmMALA";
            case MarkovChainType::DikinWalk:
                return "DW";
            case MarkovChainType::Gaussian:
                return "G";
            case MarkovChainType::HitAndRun:
                return "HR";
            default:
                throw std::runtime_error("Bug in switch case.");
        }

    }
}

#endif //HOPS_MARKOVCHAINTYPE_HPP
