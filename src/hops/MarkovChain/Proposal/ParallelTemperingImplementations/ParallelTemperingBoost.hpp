#ifndef HOPS_PARALLELTEMPERINGBOOST_HPP
#define HOPS_PARALLELTEMPERINGBOOST_HPP

#include "ParallelTempering.hpp"
#include "hops/Utility/VectorType.hpp"
#include "hops/RandomNumberGenerator/RandomNumberGenerator.hpp"

namespace hops {
    class Proposal;
    class ParallelTemperingBoost: ParallelTempering {
    public:
        ParallelTemperingBoost(int numChains, int chainIndex, const char *sharedMemoryName);

        VectorType proposeStateExchange(RandomNumberGenerator &rng, Proposal *proposal) override;

        double computeAcceptanceProbability() override;

        ~ParallelTemperingBoost() override;

    private:
        int num_chains;
        int chain_index;
        const char* sharedMemoryName;
        double otherColdness=0;
        double otherNegativeLogLikelihood=0;
    };
}


#endif //HOPS_PARALLELTEMPERINGBOOST_HPP
