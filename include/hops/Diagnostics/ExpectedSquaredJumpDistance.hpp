#ifndef HOPS_EXPECTEDSQUAREDJUMPDISTANCE_HPP
#define HOPS_EXPECTEDSQUAREDJUMPDISTANCE_HPP

#include <string>
#include <stdexcept>
#include <vector>
#include <cmath>
#include <cassert>
#include <memory>

namespace hops {
    /*
     * Compute Expected Squared Jump Distance incrementally on a single vector of draws. 
     * The Expected Squared Jump Distance is defined as
     * \[ ESJD = \frac{1}{N-1} \sum_{n=1}^(N-1) \theta_n^\top \theta_{n+1} \]
     */
    template<typename StateType>
    double computeExpectedSquaredJumpDistance (const std::vector<StateType>& draws, 
                                               unsigned long numUnseen, 
                                               double esjdSeen, 
                                               unsigned long numSeen) {
        size_t numDraws = draws.size(),
               correction = 0;
        // account for missing jump between two batches of samples
        if (numSeen > 0 && draws.size() > numUnseen) {
            correction = 1;
        }

        // in order to guarantee eta to be 1, we have to set it to 1.
        if (numSeen == 0) {
            ++numSeen;
        }

        double esjd = 0, 
               eta = 1.0 * (numSeen - 1) / (numSeen + numUnseen - 1),
               squaredDistance;
        for (unsigned long i = numDraws - numUnseen - correction; i < numDraws - 1; ++i) {
            squaredDistance = (draws[i] - draws[i+1]).adjoint() * (draws[i] - draws[i+1]) ;
            esjd += squaredDistance;
        }
        esjd /=  numUnseen - 1 + correction;
        return eta * esjdSeen + (1 - eta) * esjd;
    }

    /* 
     * Compute Expected Squared Jump Distance non-incrementally on all draws passed.
     */
    template<typename StateType>
    double computeExpectedSquaredJumpDistance (const std::vector<StateType>& draws) {
        return computeExpectedSquaredJumpDistance(draws, draws.size(), 0, 0);
    }

    /*
     * Compute Expected Squared Jump Distance for every chain in \c chains incrementally.
     */
    template<typename StateType>
    std::vector<double> computeExpectedSquaredJumpDistance (const std::vector<std::vector<StateType>>& chains, 
                                                            unsigned long numUnseen, 
                                                            std::vector<double> esjdSeen, 
                                                            unsigned long numSeen) {
        std::vector<double> esjds(chains.size());
        for (size_t i = 0; i < chains.size(); ++i) {
            esjds[i] = computeExpectedSquaredJumpDistance(chains[i], numUnseen, esjdSeen[i], numSeen);
        }
        return esjds;
    }

    /*
     * Compute Expected Squared Jump Distance non-incrementally for every chain in \c chains. 
     */
    template<typename StateType>
    std::vector<double> computeExpectedSquaredJumpDistance (const std::vector<std::vector<StateType>>& chains) {
        return computeExpectedSquaredJumpDistance(chains, chains[0].size(), std::vector<double>(chains.size()), 0); 
    }

    /*
     * Compute Expected Squared Jump Distance for every chain in \c chains incrementally. 
     * \c chains is supposed to be a vector of pointers to the actual chains.
     */
    template<typename StateType>
    std::vector<double> computeExpectedSquaredJumpDistance (const std::vector<const std::vector<StateType>*>& chains, 
                                                            unsigned long numUnseen, 
                                                            std::vector<double> esjdSeen, 
                                                            unsigned long numSeen) {
        std::vector<double> esjds(chains.size());
        for (size_t i = 0; i < chains.size(); ++i) {
            esjds[i] = computeExpectedSquaredJumpDistance(*chains[i], numUnseen, esjdSeen[i], numSeen);
        }
        return esjds;
    }

    /*
     * Compute Expected Squared Jump Distance non-incrementally for every chain in \c chains. 
     * \c chains is supposed to be a vector of pointers to the actual chains.
     */
    template<typename StateType>
    std::vector<double> computeExpectedSquaredJumpDistance (const std::vector<const std::vector<StateType>*>& chains) {
        return computeExpectedSquaredJumpDistance(chains, chains[0]->size(), std::vector<double>(chains.size()), 0); 
    }
}


#endif //HOPS_EXPECTEDSQUAREDJUMPDISTANCE_HPP

