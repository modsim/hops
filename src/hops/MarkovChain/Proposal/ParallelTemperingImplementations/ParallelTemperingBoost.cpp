#include "ParallelTemperingBoost.hpp"

#include <boost/interprocess/shared_memory_object.hpp>
#include <boost/interprocess/mapped_region.hpp>

#include "hops/MarkovChain/Proposal/Proposal.hpp"


hops::ParallelTemperingBoost::ParallelTemperingBoost(int numChains, int chainIndex, const char *sharedMemoryName)
        : num_chains(numChains), chain_index(chainIndex), sharedMemoryName(sharedMemoryName) {}

hops::ParallelTemperingBoost::~ParallelTemperingBoost() {
    boost::interprocess::shared_memory_object::remove(this->sharedMemoryName);
}

hops::VectorType
hops::ParallelTemperingBoost::proposeStateExchange(hops::RandomNumberGenerator &rng, hops::Proposal *proposal) {
    return proposal->getState();
}

double hops::ParallelTemperingBoost::computeAcceptanceProbability() {
    return 0;
}


