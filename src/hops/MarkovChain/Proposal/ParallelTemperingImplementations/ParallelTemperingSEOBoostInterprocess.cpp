#include "ParallelTemperingSEOBoostInterprocess.hpp"

#include <boost/interprocess/managed_shared_memory.hpp>
#include <boost/interprocess/sync/interprocess_condition.hpp>
#include <boost/interprocess/sync/interprocess_mutex.hpp>
#include <boost/interprocess/sync/scoped_lock.hpp>

#include "hops/MarkovChain/Proposal/Proposal.hpp"

void barrier_wait(int count, const char *sharedMemoryName, int chainIndex) {
    std::cout << "chain index waiting " << chainIndex << std::endl;
    boost::interprocess::managed_shared_memory sharedMemory{boost::interprocess::open_only, sharedMemoryName};
    auto [mutex, _mutex_size] = sharedMemory.find<boost::interprocess::interprocess_mutex>("mtx");
    auto [condition, _condition_size] = sharedMemory.find<boost::interprocess::interprocess_condition>("cnd");
    auto *should_wait = sharedMemory.find_or_construct<bool>("should_wait")(false);
    if (mutex == nullptr || condition == nullptr) {
        throw std::runtime_error(
                "mutex or condition is null. Something has gone horribly wrong. Did the main process die?");
    }
    if (--count == 0) {
        *should_wait = true;
        std::cout << "chain index notifying all " << chainIndex << std::endl;
        condition->notify_all();
        std::cout << "chain index done " << chainIndex << std::endl;
        return;
    }
    std::cout << "chain index waiting " << chainIndex << std::endl;
    {
        boost::interprocess::scoped_lock<boost::interprocess::interprocess_mutex> lock{*mutex};
        while (should_wait) {
            condition->wait(lock);
        }
    }
    std::cout << "chain index done " << chainIndex << std::endl;
}

hops::ParallelTemperingSEOBoostInterprocess::ParallelTemperingSEOBoostInterprocess(hops::RandomNumberGenerator syncRng,
                                                                                   int numChains,
                                                                                   int chainIndex,
                                                                                   const char *sharedMemoryNameSpace,
                                                                                   long numDims) :
        syncRng(std::move(syncRng)),
        numberOfChains(numChains),
        chainIndex(chainIndex),
        sharedMemoryNameSpace(sharedMemoryNameSpace) {

    if (chainIndex == 0) {
        size_t memorySize = 40024; // TODO  compute real memory footprint
//        memorySize = std::max(memorySize, size_t(512));
        std::cout << "create shared memory called '" << sharedMemoryNameSpace << "'" << std::endl;
        boost::interprocess::managed_shared_memory sharedMemory{
                boost::interprocess::open_or_create,
                this->sharedMemoryNameSpace,
                memorySize};
        std::cout << "create con" << std::endl;
        sharedMemory.find_or_construct<boost::interprocess::interprocess_condition>("cnd")();
        std::cout << "create mutex" << std::endl;
        sharedMemory.construct<boost::interprocess::interprocess_mutex>("mtx")();
        std::cout << "done initializing" << std::endl;
    }
}

hops::ParallelTemperingSEOBoostInterprocess::~ParallelTemperingSEOBoostInterprocess() {
    std::cout << "wait before destruction" << std::endl;
    barrier_wait(this->numberOfChains, this->sharedMemoryNameSpace, this->chainIndex);
    if (chainIndex == 0) {
        std::cout << "remove " << sharedMemoryNameSpace << std::endl;
        boost::interprocess::shared_memory_object::remove(sharedMemoryNameSpace);
    }
}

hops::VectorType
hops::ParallelTemperingSEOBoostInterprocess::proposeStateExchange(hops::Proposal *proposal) {
    int partnerIndex = this->findPartnerForSwap();
    std::cout << "try to propoose state" << std::endl;
    barrier_wait(numberOfChains, sharedMemoryNameSpace, chainIndex);
    std::cout << chainIndex << " wants to swap with " << partnerIndex << std::endl;
    VectorType state = proposal->getState();
    return state;
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
    if (evenOdd % 2 == 0) {
        // even communication:
        if (chainIndex % 2 == 0) {
            // this chain is even and communicates with chainIndex +1
            partnerIndex = chainIndex + 1;
        } else {
            // this chain is odd and communicates with chainIndex -1
            partnerIndex = chainIndex - 1;
        }
    } else {
        // odd communication
        if (chainIndex % 2 == 0) {
            // this chain is even and communicates with chainIndex +1
            partnerIndex = chainIndex - 1;
        } else {
            // this chain is odd and communicates with chainIndex -1
            partnerIndex = chainIndex + 1;
        }
    }

    if (numberOfChains % 2 == 0) {
        // even number of chains: you can always find a partner
        if (partnerIndex == -1) {
            partnerIndex = numberOfChains - 1;
        }
        partnerIndex = (partnerIndex) % numberOfChains;
        return partnerIndex;
    } else {
        // odd number of chains: you can NOT always find a partner
        if (partnerIndex == numberOfChains) {
            partnerIndex = -1;
        }
        return partnerIndex;
    }
}


