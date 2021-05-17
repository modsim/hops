#ifndef HOPS_POTENTIALSCALEREDUCTIONFACTOR_HPP
#define HOPS_POTENTIALSCALEREDUCTIONFACTOR_HPP

#include <string>
#include <stdexcept>
#include <vector>
#include <cmath>
#include <cassert>
#include <memory>

#include "IsConstantChain.hpp"

namespace hops {
    /*
     * states.size() is the number of chains, states[i].size() is the number of draws, states[i][j].rows() is the dimensionality
     *
     *
     */
    template<typename StateType>
    double computePotentialScaleReductionFactor (const std::vector<const std::vector<StateType>*>& chains,
                                                 unsigned long dimension,
                                                 unsigned long numUnseen,
                                                 std::vector<double>& sampleVariancesSeen,
                                                 std::vector<double>& intraChainExpectationsSeen,
                                                 double& interChainExpectationSeen,
                                                 unsigned long& numSeen) {
        using Scalar = typename StateType::Scalar;

        unsigned long d = dimension;
        unsigned long numChains = chains.size();
        assert(numChains >= 2);

        assert(numChains == intraChainExpectationsSeen.size());
        assert(numChains == sampleVariancesSeen.size());

        unsigned long numDraws = chains[0]->size();
        assert(numDraws >= numUnseen);
#ifndef NDEBUG // prevent warning
        for (auto& draws : chains) {
            assert(numDraws == draws->size());
        }
#endif // NDEBUG

        if (isConstantChain(chains, dimension)) {
            return std::numeric_limits<double>::quiet_NaN();
        }

        auto intraChainExpectationsUnseen = std::vector<double>(numChains, 0),
             intraChainExpectations = std::vector<double>(numChains, 0);
        double interChainExpectation,
               interChainExpectationUnseen = 0,
               eta = numSeen / double((numUnseen + numSeen));
        for (unsigned long m = 0; m < numChains; ++m) {
            for (unsigned long n = numDraws - numUnseen; n < numDraws; ++n) {
                intraChainExpectationsUnseen[m] += (*chains[m])[n](d) / double(numUnseen);
            }
            intraChainExpectations[m] = eta * intraChainExpectationsSeen[m] + (1 - eta) * intraChainExpectationsUnseen[m]; 
            interChainExpectationUnseen += intraChainExpectationsUnseen[m] / numChains;
        }
      
        interChainExpectation = eta * interChainExpectationSeen + (1 - eta) * interChainExpectationUnseen;
             
        auto sampleVariances = std::vector<double>(numChains, 0);
        Scalar betweenChainVariance = 0,
               withinChainVariance = 0,
               biasedVariance = 0,
               kappa = (numUnseen + numSeen - 1);  

        for (unsigned long m = 0; m < numChains; ++m) {
            betweenChainVariance += (numSeen + numUnseen) * std::pow(intraChainExpectations[m] - interChainExpectation, 2) / (numChains - 1); 
            biasedVariance = 0;
            for (unsigned long n = numDraws - numUnseen; n < numDraws; ++n) {
                biasedVariance += std::pow((*chains[m])[n](d) - intraChainExpectations[m], 2);
            }
            sampleVariances[m] = 
                ( (numSeen - 1) * sampleVariancesSeen[m] +
                numSeen * std::pow(intraChainExpectationsSeen[m] - intraChainExpectations[m], 2) + 
                biasedVariance ) / kappa ;

            withinChainVariance += sampleVariances[m] / numChains;
        }

        Scalar rHat = std::sqrt(((numUnseen + numSeen - 1) + betweenChainVariance / withinChainVariance) / (numSeen + numUnseen));

        // update seen values
        sampleVariancesSeen = sampleVariances;
        intraChainExpectationsSeen = intraChainExpectations;
        interChainExpectationSeen = interChainExpectation;
        numSeen = numSeen + numUnseen;

        return rHat;
    }

    /*
     * states.size() is the number of chains, states[i].size() is the number of draws, states[i][j].rows() is the dimensionality
     *
     *
     */
    template<typename StateType>
    double computePotentialScaleReductionFactor (const std::vector<std::vector<StateType>>& chains,
                                                 unsigned long dimension,
                                                 unsigned long numUnseen,
                                                 std::vector<double>& sampleVariancesSeen,
                                                 std::vector<double>& intraChainExpectationsSeen,
                                                 double& interChainExpectationSeen,
                                                 unsigned long& numSeen) {
        std::vector<const std::vector<StateType>*> chainsPtrArray;
        for (auto& chain : chains) {
            chainsPtrArray.push_back(&chain);
        }
        return computePotentialScaleReductionFactor(chainsPtrArray, 
                                                    dimension, 
                                                    numUnseen, 
                                                    sampleVariancesSeen, 
                                                    intraChainExpectationsSeen, 
                                                    interChainExpectationSeen, 
                                                    numSeen);
    }

    /*
     * Compute PSRF for all dimensions at once.
     *
     *
     */
    template<typename StateType>
    std::vector<double> computePotentialScaleReductionFactor (const std::vector<const std::vector<StateType>*>& chains,
                                                              unsigned long numUnseen,
                                                              std::vector<std::vector<double>>& sampleVariancesSeen,
                                                              std::vector<std::vector<double>>& intraChainExpectationsSeen,
                                                              std::vector<double>& interChainExpectationSeen,
                                                              unsigned long& numSeen) {
        unsigned long dimensions = (*chains[0])[0].rows();
        unsigned long numChains = chains.size();

        if (numChains <= 1) {
            throw std::runtime_error(std::string("PSRF is only defined for at least two chains, your data consists of only ") + std::to_string(numChains));
        }

        // initialize empty vectors correctly
        if (sampleVariancesSeen.size() == 0) {
            sampleVariancesSeen = std::vector<std::vector<double>>(dimensions,
                                                                   std::vector<double>(numChains, 0));
        }

        if (intraChainExpectationsSeen.size() == 0) {
            intraChainExpectationsSeen = std::vector<std::vector<double>>(dimensions,
                                                                          std::vector<double>(numChains, 0));
        }

        if (interChainExpectationSeen.size() == 0) {
            interChainExpectationSeen = std::vector<double>(dimensions, 0);
        }

        assert(dimensions == sampleVariancesSeen.size());
        assert(dimensions == intraChainExpectationsSeen.size());
        assert(dimensions == interChainExpectationSeen.size());

        unsigned long numSeenUpdated;

        std::vector<double> rhats = std::vector<double>(dimensions); 
        for (unsigned long dimension = 0; dimension < dimensions; ++dimension) {
            unsigned long _numSeen = numSeen;
            rhats[dimension] = computePotentialScaleReductionFactor(chains, 
                                                                    dimension,
                                                                    numUnseen, 
                                                                    sampleVariancesSeen[dimension], 
                                                                    intraChainExpectationsSeen[dimension],
                                                                    interChainExpectationSeen[dimension],
                                                                    _numSeen);
            numSeenUpdated = _numSeen;
        }

        numSeen = numSeenUpdated;

        return rhats;
    }

    /*
     * Compute PSRF for all dimensions at once.
     *
     *
     */
    template<typename StateType>
    std::vector<double> computePotentialScaleReductionFactor (const std::vector<std::vector<StateType>>& chains,
                                                              unsigned long numUnseen,
                                                              std::vector<std::vector<double>>& sampleVariancesSeen,
                                                              std::vector<std::vector<double>>& intraChainExpectationsSeen,
                                                              std::vector<double>& interChainExpectationSeen,
                                                              unsigned long& numSeen) {
        std::vector<const std::vector<StateType>*> chainsPtrArray;
        for (auto& chain : chains) {
            chainsPtrArray.push_back(&chain);
        }
        return computePotentialScaleReductionFactor(chainsPtrArray, 
                                                    numUnseen, 
                                                    sampleVariancesSeen, 
                                                    intraChainExpectationsSeen, 
                                                    interChainExpectationSeen, 
                                                    numSeen);
    }

    template<typename StateType>
    double computePotentialScaleReductionFactor (const std::vector<const std::vector<StateType>*>& chains,
                                                 unsigned long dimension) {
        std::vector<double> sampleVariances(chains.size());
        std::vector<double> intraChainExpectations(chains.size());
        double interChainExpectation = 0;
        unsigned long numSeen = 0;
        return computePotentialScaleReductionFactor(chains, 
                                                    dimension, 
                                                    chains[0]->size(),
                                                    sampleVariances,
                                                    intraChainExpectations,
                                                    interChainExpectation,
                                                    numSeen);
    }

    template<typename StateType>
    double computePotentialScaleReductionFactor (const std::vector<std::vector<StateType>>& chains, 
                                                 unsigned long dimension) {
        std::vector<const std::vector<StateType>*> chainsPtrArray;
        for (auto& chain : chains) {
            chainsPtrArray.push_back(&chain);
        }
        return computePotentialScaleReductionFactor(chainsPtrArray, dimension);
    }

    template<typename StateType>
    std::vector<double> computePotentialScaleReductionFactor (const std::vector<const std::vector<StateType>*>& chains) {
        unsigned long dimensions = (*chains[0])[0].rows();
        std::vector<double> rhats = std::vector<double>(dimensions); 
        for (unsigned long dimension = 0; dimension < dimensions; ++dimension) {
            rhats[dimension] = computePotentialScaleReductionFactor(chains, dimension);
        }
        return rhats;
    }

    template<typename StateType>
    std::vector<double> computePotentialScaleReductionFactor (const std::vector<std::vector<StateType>>& chains) {
        std::vector<const std::vector<StateType>*> chainsPtrArray;
        for (auto& chain : chains) {
            chainsPtrArray.push_back(&chain);
        }
        return computePotentialScaleReductionFactor(chainsPtrArray);
    }
}


#endif //HOPS_POTENTIALSCALEREDUCTIONFACTOR_HPP

