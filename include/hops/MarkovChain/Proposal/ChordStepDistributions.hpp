#ifndef HOPS_CHORDSTEPDISTRIBUTIONS_HPP
#define HOPS_CHORDSTEPDISTRIBUTIONS_HPP

#include <hops/RandomNumberGenerator/RandomNumberGenerator.hpp>
#include <random>
#include <string>
#include "TruncatedNormalDistribution.hpp"

namespace hops {
    template<typename RealType = double>
    class UniformStepDistribution {
    public:
        RealType draw(RandomNumberGenerator &randomNumberGenerator, RealType lowerLimit, RealType upperLimit) {
            typename std::uniform_real_distribution<RealType>::param_type params(lowerLimit, upperLimit);
            return uniformRealDistribution(randomNumberGenerator, params);
        }

        constexpr RealType calculateInverseNormalizationConstant(RealType, RealType, RealType) {
            return 1.;
        }

    private:
        std::uniform_real_distribution<RealType> uniformRealDistribution;
    };

    template<typename RealType = double>
    class GaussianStepDistribution {
    public:
        RealType draw(RandomNumberGenerator &randomNumberGenerator, RealType lowerLimit,
                      RealType upperLimit) {
            return truncatedNormalDistribution(randomNumberGenerator, {stepSize, lowerLimit, upperLimit});
        }

        RealType getStepSize(RealType newStepSize) {
            stepSize = newStepSize;
        }

        void setStepSize(RealType newStepSize) {
            stepSize = newStepSize;
        }

        RealType calculateInverseNormalizationConstant(RealType sigma, RealType lowerBound, RealType upperBound) {
            return truncatedNormalDistribution.inverseNormalization({sigma, lowerBound, upperBound});
        }

    private:
        RealType stepSize = 1.;
        TruncatedNormalDistribution<RealType> truncatedNormalDistribution;
    };
}

#endif //HOPS_CHORDSTEPDISTRIBUTIONS_HPP
