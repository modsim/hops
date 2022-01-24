#ifndef HOPS_MARKOVCHAINTYPE_HPP
#define HOPS_MARKOVCHAINTYPE_HPP

#include <vector>

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
    };

    std::string markovChainTypeToFullString(MarkovChainType markovChainType) {
        switch (markovChainType) {
            case MarkovChainType::AdaptiveMetropolis:
                return "Adaptive Metropolis";
            case MarkovChainType::BallWalk:
                return "Ball Walk";
            case MarkovChainType::BilliardAdaptiveMetropolis:
                return "Billiard Adaptive Metropolis";
            case MarkovChainType::BilliardMALA:
                return "Billiard MALA";
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
                throw std::runtime_error("Bug in switch case for markovChainTypeToFullString.");
        }
    }

    std::string markovChainTypeToShortString(MarkovChainType markovChainType) {
        switch (markovChainType) {
            case MarkovChainType::AdaptiveMetropolis:
                return "AM";
            case MarkovChainType::BallWalk:
                return "BW";
            case MarkovChainType::BilliardAdaptiveMetropolis:
                return "BAM";
            case MarkovChainType::BilliardMALA:
                return "BMALA";
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
                throw std::runtime_error("Bug in switch case for markovChainTypeToShortString.");
        }
    }

    bool checkIfStringIsEqualToChainType(const std::string &chainTypeString, MarkovChainType markovChainType) {
        std::string shortStringType = markovChainTypeToShortString(markovChainType);
        std::string longStringType = markovChainTypeToFullString(markovChainType);
        return chainTypeString == shortStringType || chainTypeString == longStringType;
    }

    MarkovChainType stringToMarkovChainType(const std::string &chainString) {
        std::vector<MarkovChainType> chainTypes{
                MarkovChainType::AdaptiveMetropolis,
                MarkovChainType::BallWalk,
                MarkovChainType::BilliardAdaptiveMetropolis,
                MarkovChainType::BilliardMALA,
                MarkovChainType::CoordinateHitAndRun,
                MarkovChainType::CSmMALA,
                MarkovChainType::DikinWalk,
                MarkovChainType::Gaussian,
                MarkovChainType::HitAndRun
        };

        for (const MarkovChainType &chainType: chainTypes) {
            if (checkIfStringIsEqualToChainType(chainString, chainType)) {
                return chainType;
            }
        }
        throw std::invalid_argument(chainString + " can not be converted to valid MarkovChainType.");
    }
}

#endif //HOPS_MARKOVCHAINTYPE_HPP
