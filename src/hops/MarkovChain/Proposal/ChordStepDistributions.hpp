#ifndef HOPS_CHORDSTEPDISTRIBUTIONS_HPP
#define HOPS_CHORDSTEPDISTRIBUTIONS_HPP

#include "../../RandomNumberGenerator/RandomNumberGenerator.hpp"
#include <random>
#include <string>
#include "TruncatedNormalDistribution.hpp"

namespace hops {
    template<typename RealType = double>
    class UniformStepDistribution {
    public:
        RealType draw(RandomNumberGenerator &randomNumberGenerator, RealType lowerLimit, RealType upperLimit) {
            if(lowerLimit > upperLimit) {
                throw std::invalid_argument("UniformStepDistribution: Lower limit is larger than upper limit. Check the polytope.");
            }
            if(lowerLimit <= std::numeric_limits<RealType>::lowest() || upperLimit >= std::numeric_limits<RealType>::max()) {
                throw std::invalid_argument("UniformStepDistribution: Upper (" + std::to_string(upperLimit) + ") or lower limit (" + std::to_string(lowerLimit) + ") is unconstrained, therefore the draw is not well-defined. Try constraining the polytope with upper and lower bounds.");
            }
            typename std::uniform_real_distribution<RealType>::param_type params(lowerLimit, upperLimit);
            auto val = uniformRealDistribution(randomNumberGenerator, params);
            return val;
        }

        constexpr RealType computeInverseNormalizationConstant(RealType, RealType, RealType) {
            return 1.;
        }

        RealType computeProbabilityDensity(RealType x, RealType lowerLimit, RealType upperLimit){
            if(lowerLimit >= upperLimit) {
                throw std::invalid_argument("GaussianStepDistribution: Lower limit is larger than upper limit. Check the polytope.");
            }
            if (lowerLimit < x && x < upperLimit) {
                return 1./(upperLimit - lowerLimit);
            }
            return 0.;
        }

    private:
        std::uniform_real_distribution<RealType> uniformRealDistribution;
    };

    template<typename RealType = double>
    class GaussianStepDistribution {
    public:
        RealType draw(RandomNumberGenerator &randomNumberGenerator, RealType lowerLimit,
                      RealType upperLimit) {
            if(lowerLimit >= upperLimit) {
                throw std::invalid_argument("GaussianStepDistribution: Lower limit is larger than upper limit. Check the polytope.");
            }
            return truncatedNormalDistribution(randomNumberGenerator, {stepSize, lowerLimit, upperLimit});
        }

        RealType draw(RandomNumberGenerator &randomNumberGenerator,
                          double sigma,
                          RealType lowerLimit,
                          RealType upperLimit) {
            if(lowerLimit >= upperLimit) {
                throw std::invalid_argument("GaussianStepDistribution: Lower limit is larger than upper limit. Check the polytope.");
            }
            return truncatedNormalDistribution(randomNumberGenerator, {sigma, lowerLimit, upperLimit});
        }

        RealType getStepSize() const {
            return stepSize;
        }

        void setStepSize(RealType newStepSize) {
            stepSize = newStepSize;
        }

        RealType computeInverseNormalizationConstant(RealType sigma, RealType lowerLimit, RealType upperLimit) {
            if(lowerLimit >= upperLimit) {
                throw std::invalid_argument("GaussianStepDistribution: Lower limit is larger than upper limit. Check the polytope.");
            }
            return truncatedNormalDistribution.inverseNormalization({sigma, lowerLimit, upperLimit});
        }

        RealType computeProbabilityDensity(RealType x, RealType sigma, RealType lowerLimit, RealType upperLimit){
            if(lowerLimit >= upperLimit) {
                throw std::invalid_argument("GaussianStepDistribution: Lower limit is larger than upper limit. Check the polytope.");
            }
            return truncatedNormalDistribution.probabilityDensity(x, sigma, lowerLimit, upperLimit);
        }

    private:
        RealType stepSize = 1.;
        TruncatedNormalDistribution<RealType> truncatedNormalDistribution;
    };
}

#endif //HOPS_CHORDSTEPDISTRIBUTIONS_HPP
