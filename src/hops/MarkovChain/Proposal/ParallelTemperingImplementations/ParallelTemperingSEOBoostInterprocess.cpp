#include "ParallelTemperingSEOBoostInterprocess.hpp"

#include <boost/interprocess/shared_memory_object.hpp>
#include <boost/interprocess/mapped_region.hpp>

#include "hops/MarkovChain/Proposal/Proposal.hpp"


hops::ParallelTemperingSEOBoostInterprocess::ParallelTemperingSEOBoostInterprocess(hops::RandomNumberGenerator syncRng,
                                                                                   int numChains,
                                                                                   int chainIndex,
                                                                                   const char *sharedMemoryName) :
        syncRng(std::move(syncRng)),
        numberOfChains(numChains),
        chainIndex(chainIndex),
        sharedMemoryName(sharedMemoryName) {

//    boost::interprocess::shared_memory_object shm_obj(
//            boost::interprocess::open_or_create,
//            shared_memory_name,
//            boost::interprocess::read_write);
}

hops::ParallelTemperingSEOBoostInterprocess::~ParallelTemperingSEOBoostInterprocess() {
    boost::interprocess::shared_memory_object::remove(this->sharedMemoryName);
}

hops::VectorType
hops::ParallelTemperingSEOBoostInterprocess::proposeStateExchange(hops::RandomNumberGenerator &rng, hops::Proposal *proposal) {
    int partnerIndex = this->findPartnerForSwap();

    return proposal->getState();
}

double hops::ParallelTemperingSEOBoostInterprocess::computeAcceptanceProbability() {
    return 0;
}

std::string hops::ParallelTemperingSEOBoostInterprocess::getName() const {
    return "Parallel Tempering with Boost.Interprocess";
}

int hops::ParallelTemperingSEOBoostInterprocess::findPartnerForSwap() {
    int partnerIndex = -1;
    int evenOdd = this->uniformIntDistribution(syncRng, std::uniform_int_distribution<int>::param_type(0, 1));
    if(evenOdd % 2 == 0) {
        // even communication:
        if(chainIndex % 2 == 0) {
            // this chain is even and communicates with chainIndex +1
            partnerIndex= chainIndex + 1;
        }
        else {
            // this chain is odd and communicates with chainIndex -1
            partnerIndex= chainIndex - 1;
        }
    }
    else {
        // odd communication
        if(chainIndex % 2 == 0) {
            // this chain is even and communicates with chainIndex +1
            partnerIndex= chainIndex - 1;
        }
        else {
            // this chain is odd and communicates with chainIndex -1
            partnerIndex= chainIndex + 1;
        }
    }

    if(numberOfChains % 2 == 0) {
        // even number of chains: you can always find a partner
        if(partnerIndex==-1) {
            partnerIndex = numberOfChains - 1;
        }
        partnerIndex = (partnerIndex) % numberOfChains;
        return partnerIndex;
    }
    else {
        // odd number of chains: you can NOT always find a partner
        if(partnerIndex == numberOfChains) {
            partnerIndex = -1;
        }
        return partnerIndex;
    }
}


