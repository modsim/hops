#include "ParallelTemperingSEOBoostInterprocess.hpp"

#include <boost/interprocess/managed_shared_memory.hpp>
#include <boost/interprocess/sync/named_mutex.hpp>
#include "hops/MarkovChain/Proposal/Proposal.hpp"


hops::ParallelTemperingSEOBoostInterprocess::ParallelTemperingSEOBoostInterprocess(hops::RandomNumberGenerator syncRng,
                                                                                   int numChains,
                                                                                   int chainIndex,
                                                                                   const char *sharedMemoryNameSpace) :
        syncRng(std::move(syncRng)),
        numberOfChains(numChains),
        chainIndex(chainIndex),
        sharedMemoryNameSpace(sharedMemoryNameSpace) {}

hops::ParallelTemperingSEOBoostInterprocess::~ParallelTemperingSEOBoostInterprocess() {
    boost::interprocess::named_mutex ourMutex{boost::interprocess::open_or_create, std::to_string(chainIndex).c_str()};
    std::string ourName = std::string(sharedMemoryNameSpace) + std::to_string(chainIndex);
    std::cout << "locking mutex for chain " << chainIndex << std::endl;
//    interprocess_mutex *mtx = managed_shm.find_or_construct<interprocess_mutex>("mtx")();
//    ourMutex.lock();
    std::cout << "remove shared memory region " << chainIndex << std::endl;
    boost::interprocess::shared_memory_object::remove(ourName.c_str());
//    ourMutex.unlock();
    std::cout << "done " << chainIndex << std::endl;
}

hops::VectorType
hops::ParallelTemperingSEOBoostInterprocess::proposeStateExchange(hops::Proposal *proposal) {
    std::cout << chainIndex << " wtf0" << std::endl;
    int partnerIndex = this->findPartnerForSwap();

    std::cout << chainIndex << " wtf1" << std::endl;
    std::string ourName = std::string(sharedMemoryNameSpace) + std::to_string(chainIndex);
    std::cout << chainIndex << " wtf2" << std::endl;

    VectorType state = proposal->getState();
    std::cout << chainIndex << " wtf3" << std::endl;
    boost::interprocess::named_mutex ourMutex{boost::interprocess::open_or_create, std::to_string(chainIndex).c_str()};
    std::cout << std::endl << chainIndex << " wtf4" << std::endl;
    ourMutex.lock();
    std::cout << chainIndex << " wtf5" << std::endl;
    std::cout << chainIndex << " creating memory of size " << sizeof(state) << std::endl;
    boost::interprocess::managed_shared_memory ourMemory{
        boost::interprocess::open_or_create,
        ourName.c_str(),
        sizeof(state)
    };
    std::cout << std::endl << chainIndex << " wtf6" << std::endl;
    ourMemory.construct<VectorType>("State")(state);
    ourMutex.unlock();
    std::cout << std::endl << chainIndex << " wtf7" << std::endl;
    std::cout << std::endl << chainIndex << " wtf8" << std::endl;

    std::string theirName = std::string(sharedMemoryNameSpace) + std::to_string(partnerIndex);
    std::cout << std::endl << chainIndex << " wtf9" << std::endl;
    boost::interprocess::named_mutex theirMutex{boost::interprocess::open_or_create, std::to_string(partnerIndex).c_str()};
    boost::interprocess::managed_shared_memory theirMemory{
            boost::interprocess::open_or_create,
            theirName.c_str(),
            sizeof(state)
    };
    std::cout << std::endl << chainIndex << " wtf8" << std::endl;
    theirMutex.lock();
    std::pair<VectorType*, std::size_t> theirState = theirMemory.find<VectorType>("State");
    theirMutex.unlock();
    boost::interprocess::named_mutex printmutex{boost::interprocess::open_or_create, "printmutex"};
    printmutex.lock();
    std::cout << "chain " << chainIndex << " sent " << state.transpose() << " and got " << theirState.first->transpose() << std::endl;
    printmutex.unlock();
    std::cout << "wtf " << std::endl;
    return *theirState.first;
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


