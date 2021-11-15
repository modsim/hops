#ifndef ISCONSTANTCHAIN_HPP
#define ISCONSTANTCHAIN_HPP

#include <stdexcept>
#include <vector>
#include <cmath>
#include <cassert>
#include <memory>
#include <limits>

namespace hops {
    // taken from https://en.cppreference.com/w/cpp/types/numeric_limits/epsilon
    template<class T>
    typename std::enable_if<!std::numeric_limits<T>::is_integer, bool>::type
    almost_equal(T x, T y, int ulp = 2)
    {
        // the machine epsilon has to be scaled to the magnitude of the values used
        // and multiplied by the desired precision in ULPs (units in the last place)
        return std::fabs(x-y) <= std::numeric_limits<T>::epsilon() * std::fabs(x+y) * ulp
            // unless the result is subnormal
            || std::fabs(x-y) < std::numeric_limits<T>::min();
    }

    template<typename StateType>
    bool isConstantChain (const std::vector<const std::vector<StateType>*>& chains, long dimension) {
        using Scalar = typename StateType::Scalar;

        long d = dimension;
        unsigned long numberOfChains = chains.size();
        unsigned long numberOfDraws = chains[0]->size();
       
        std::vector<Scalar> initDraw(numberOfChains, 0);

        bool chainsAllConst = true;
        for (unsigned long m = 0; m < numberOfChains; ++m) {
            initDraw[m] = (*chains[m])[0](d);

            if (!std::isfinite(initDraw[m])) {
                return std::numeric_limits<double>::quiet_NaN();
            }

            // check if all draws are (almost) equal to the first value.
            // if one is different, break.
            bool drawsAllConst = true;
            for (unsigned long n = 1; n < numberOfDraws; ++n) {
                if (!std::isfinite((*chains[m])[n](d))) {
                    return true;
                }

                if (!almost_equal(initDraw[m], (*chains[m])[n](d))) {
                    drawsAllConst = false;
                    break;
                }
            }

            // record if any of the chains has draws which are not all const
            if (!drawsAllConst) {
                chainsAllConst = false;
                break;
            }
        }

        if (chainsAllConst) {
            bool chainsAllSameConst = true;
            // If all chains are constant then return NaN
            // if they all equal the same constant value
            for (unsigned long m = 1; m < numberOfChains; ++m) {
                if (!almost_equal(initDraw[0], initDraw[m])) {
                    chainsAllSameConst = false;
                    break;
                }
            }

            if (chainsAllSameConst) {
                return true;
            }
        }

        return false;
    }

    template<typename StateType>
    bool isConstantChain (const std::vector<std::vector<StateType>>& chains, long dimension) {
        std::vector<const std::vector<StateType>*> chainsPtrArray;
        for (auto& chain : chains) {
            chainsPtrArray.push_back(&chain);
        }
        return isConstantChain(chainsPtrArray, dimension);
    }

    template<typename StateType>
    bool isConstantChain (const std::vector<const std::vector<StateType>*>& chains) {
        for (long dimension = 0; dimension < (*chains[0])[0].size(); ++dimension) {
            if (!isConstantChain(chains, dimension)) {
                return false;
            }
        }

        return true;
    }

    template<typename StateType>
    bool isConstantChain (const std::vector<std::vector<StateType>>& chains) {
        std::vector<const std::vector<StateType>*> chainsPtrArray;
        for (auto& chain : chains) {
            chainsPtrArray.push_back(&chain);
        }
        return isConstantChain(chainsPtrArray);
    }
}

#endif // ISCONSTANTCHAIN_HPP
