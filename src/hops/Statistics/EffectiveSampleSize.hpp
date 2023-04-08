#ifndef HOPS_EFFECTIVESAMPLESIZE_HPP
#define HOPS_EFFECTIVESAMPLESIZE_HPP

#include <vector>
#include <iostream>
#include <memory>

#include "Autocorrelation.hpp"

namespace hops {
    using IntermediateEffectiveSampleSizeResults = double;

    /**
     * @deprecated
     * @tparam StateType
     * @param chains
     * @param dimension
     * @return
     */
    template<typename StateType>
    double computeEffectiveSampleSize(const std::vector<const std::vector<StateType> *> &chains,
                                      unsigned long dimension) {
        unsigned long d = dimension;
        unsigned long numChains = chains.size();
        if (numChains == 0) {
            throw std::invalid_argument("No chains. Cannot compute ESS.");
        }
        unsigned long numDraws = chains[0]->size();
        if (numDraws == 0) {
            throw std::invalid_argument("No samples in chains. Cannot compute ESS.");
        }

        double interChainExpectation = 0;
        double betweenChainVariance = 0;
        double withinChainVariance = 0;
        double varianceEstimate = 0;

        std::vector<double> rhoHat;
        std::vector<double> intraChainExpectations = std::vector<double>(numChains, 0);
        std::vector<double> sampleVariances = std::vector<double>(numChains, 0);

        std::vector<Eigen::VectorXd> autocorrelations(numChains);

        for (size_t m = 0; m < numChains; ++m) {
            for (size_t n = 0; n < numDraws; ++n) {
                intraChainExpectations[m] += (*chains[m])[n](d);
            }
            interChainExpectation += intraChainExpectations[m];
            intraChainExpectations[m] /= numDraws;
        }

        interChainExpectation /= numChains * numDraws;

        for (size_t m = 0; m < numChains; ++m) {
            betweenChainVariance += std::pow(intraChainExpectations[m] - interChainExpectation, 2);
            for (size_t n = 0; n < numDraws; ++n) {
                sampleVariances[m] += std::pow((*chains[m])[n](d) - intraChainExpectations[m], 2);
            }
            withinChainVariance += sampleVariances[m];
            sampleVariances[m] /= numDraws - 1;
        }

        // between-chain variance is not defined for single chains
        if (numChains > 1) {
            betweenChainVariance *= numDraws;
            betweenChainVariance /= numChains - 1;
        } else {
            betweenChainVariance = 0;
        }
        withinChainVariance /= (numDraws - 1) * numChains;
        varianceEstimate = ((numDraws - 1) * withinChainVariance + betweenChainVariance) / numDraws;

        for (size_t m = 0; m < numChains; ++m) {
            computeAutocorrelations(chains[m], autocorrelations[m], d);
        }

        double rhoHatEven, rhoHatOdd, autocovariance;
        for (size_t t = 0; t < numDraws / 2; ++t) {
            autocovariance = 0;
            for (size_t m = 0; m < numChains; ++m) {
                autocovariance += (numDraws - 1) * sampleVariances[m] * autocorrelations[m](2 * t);
            }
            autocovariance /= numChains * numDraws;

            //std::cout << "1 - (" << withinChainVariance << " - " << autocovariance << ") / " << varianceEstimate << std::endl;

            // according to stan implementation, set first rhohat to 1.
            rhoHatEven = (t == 0 ? 1 : 1 - (withinChainVariance - autocovariance) / varianceEstimate);

            autocovariance = 0;
            for (size_t m = 0; m < numChains; ++m) {
                autocovariance += (numDraws - 1) * sampleVariances[m] * autocorrelations[m](2 * t + 1);
            }
            autocovariance /= numChains * numDraws;

            //std::cout << "1 - (" << withinChainVariance << " - " << autocovariance << ") / " << varianceEstimate << std::endl;

            rhoHatOdd = 1 - (withinChainVariance - autocovariance) / varianceEstimate;

            // break if sequence becomes negative
            if (rhoHatEven + rhoHatOdd <= 0) {
                break;
            }

            rhoHat.push_back(rhoHatEven);
            rhoHat.push_back(rhoHatOdd);
        }

        if (rhoHatEven > 0) {
            rhoHat.push_back(rhoHatEven);
        }

        double tauHat = -1;

        // turn initial positive sequence to initial monotone sequence, as in stan implementation.
        // TODO: find detailed explanation of how to do this in some paper.
        for (size_t t = 1; t < (rhoHat.size() - 2) / 2; ++t) {
            if (rhoHat[2 * t] + rhoHat[2 * t + 1] > rhoHat[2 * t - 2] + rhoHat[2 * t - 1]) {
                rhoHat[2 * t] = (rhoHat[2 * t - 2] + rhoHat[2 * t - 1]) / 2;
                rhoHat[2 * t + 1] = rhoHat[2 * t];
            }
        }

        for (size_t t = 0; t < rhoHat.size() / 2; ++t) {
            tauHat += 2 * (rhoHat[2 * t] + rhoHat[2 * t + 1]);
        }

        // we can only have odd number of rhoHats, if we added the antiethic case improvement
        if (rhoHat.size() % 2) {
            tauHat += rhoHat.back();
        }


        // according to Vehtari et al. 2020
        return std::min(numDraws * numChains / tauHat, numDraws * numChains * std::log10(numDraws * numChains));
    }

    /**
     * @deprecated
     * @tparam StateType
     * @param chains
     * @param dimension
     * @return
     */
    template<typename StateType>
    std::vector<double> computeEffectiveSampleSize(const std::vector<const std::vector<StateType> *> &chains) {
        std::vector<double> ess;
        for (long d = 0; d < (*chains[0])[0].size(); ++d) {
            ess.push_back(computeEffectiveSampleSize(chains, d));
        }

        return ess;
    }

    /**
     * @deprecated
     * @tparam StateType
     * @param chains
     * @param dimension
     * @return
     */
    template<typename StateType>
    double computeEffectiveSampleSize(const std::vector<std::vector<StateType>> &chains, unsigned long dimension) {
        std::vector<const std::vector<StateType> *> chainsPtrArray;
        for (auto &chain : chains) {
            chainsPtrArray.push_back(&chain);
        }
        return computeEffectiveSampleSize<StateType>(chainsPtrArray, dimension);
    }

    /**
     * @deprecated
     * @tparam StateType
     * @param chains
     * @param dimension
     * @return
     */
    template<typename StateType>
    std::vector<double> computeEffectiveSampleSize(const std::vector<std::vector<StateType>> &chains) {
        std::vector<const std::vector<StateType> *> chainsPtrArray;
        for (auto &chain : chains) {
            chainsPtrArray.push_back(&chain);
        }
        return computeEffectiveSampleSize<StateType>(chainsPtrArray);
    }
}

#endif // HOPS_EFFECTIVESAMPLESIZE_HPP
