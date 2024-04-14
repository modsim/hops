#ifndef HOPS_PARALLELTEMPERINGSEOBOOSTINTERPROCESS_HPP
#define HOPS_PARALLELTEMPERINGSEOBOOSTINTERPROCESS_HPP

#include <random>

#include "ParallelTempering.hpp"
#include "hops/Utility/VectorType.hpp"
#include "hops/RandomNumberGenerator/RandomNumberGenerator.hpp"

namespace hops {
    class Proposal;
    class ParallelTemperingSEOBoostInterprocess: ParallelTempering {
    public:
        ParallelTemperingSEOBoostInterprocess(RandomNumberGenerator syncRng, int numChains, int chainIndex, const char *sharedMemoryName);

        VectorType proposeStateExchange(RandomNumberGenerator &rng, Proposal *proposal) override;

        double computeAcceptanceProbability() override;

        [[nodiscard]] std::string getName() const override;

        int findPartnerForSwap();

        ~ParallelTemperingSEOBoostInterprocess() override;

        RandomNumberGenerator syncRng;
        int numberOfChains;
        const int chainIndex;
        const char* sharedMemoryName;
        double otherColdness=0;
        double otherNegativeLogLikelihood=0;

    private:
        std::uniform_int_distribution<int> uniformIntDistribution;
        std::uniform_real_distribution<double> uniformRealDistribution;
    };
}


#endif //HOPS_PARALLELTEMPERINGSEOBOOSTINTERPROCESS_HPP
