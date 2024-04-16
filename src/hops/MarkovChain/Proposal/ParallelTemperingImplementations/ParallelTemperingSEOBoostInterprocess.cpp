#include "ParallelTemperingSEOBoostInterprocess.hpp"

#include <boost/interprocess/managed_shared_memory.hpp>
#include <boost/interprocess/sync/interprocess_condition.hpp>
#include <boost/interprocess/sync/interprocess_mutex.hpp>
#include <boost/interprocess/sync/scoped_lock.hpp>

#include "hops/MarkovChain/Proposal/Proposal.hpp"


hops::ParallelTemperingSEOBoostInterprocess::ParallelTemperingSEOBoostInterprocess(hops::RandomNumberGenerator syncRng,
                                                                                   int numChains,
                                                                                   int chainIndex,
                                                                                   const char *sharedMemoryNameSpace,
                                                                                   long numDims) :
        syncRng(std::move(syncRng)),
        numberOfChains(numChains),
        chainIndex(chainIndex),
        sharedMemoryNameSpace(sharedMemoryNameSpace) {

    size_t memorySize = static_cast<std::size_t>(numDims * sizeof(hops::VectorType::Scalar)) +
                        2 * sizeof(boost::interprocess::interprocess_condition);
    memorySize = std::max(memorySize, size_t(512));

    std::string name = this->getSharedMemoryName(chainIndex);
    std::cout << "create shared memory with " << name << std::endl;

    boost::interprocess::shared_memory_object::remove(name.c_str());
    boost::interprocess::managed_shared_memory ourMemory{
            boost::interprocess::open_or_create,
            name.c_str(),
            memorySize};

    if (chainIndex == 0) {
        ourMemory.construct<boost::interprocess::interprocess_condition>("barrier")();
    }
}

hops::ParallelTemperingSEOBoostInterprocess::~ParallelTemperingSEOBoostInterprocess() {
    std::string name = this->getSharedMemoryName(chainIndex);
//    boost::interprocess::managed_shared_memory ourMemory{
//            boost::interprocess::open_only,
//            name.c_str()
//    };
//    boost::interprocess::interprocess_condition *interprocess_condition =
//            ourMemory.find<boost::interprocess::interprocess_condition>("barrier").first;
//    boost::interprocess::interprocess_mutex *mtx =
//            ourMemory.find_or_construct<boost::interprocess::interprocess_mutex>("mtx")();
//    boost::interprocess::scoped_lock<boost::interprocess::interprocess_mutex> lock{*mtx};
//    interprocess_condition->notify_all();
//    std::cout << chainIndex << " is waiting" << std::endl;
//    interprocess_condition->wait(lock);
//    std::cout << chainIndex << " waited" << std::endl;
//    interprocess_condition->notify_all();
    std::cout << "remove " << name << std::endl;
//    boost::interprocess::shared_memory_object::remove(name.c_str());
}

hops::VectorType
hops::ParallelTemperingSEOBoostInterprocess::proposeStateExchange(hops::Proposal *proposal) {
    int partnerIndex = this->findPartnerForSwap();

    std::string mainName = this->getSharedMemoryName(0);
    std::string ourName = this->getSharedMemoryName(chainIndex);
    std::string partnerName = this->getSharedMemoryName(partnerIndex);

    boost::interprocess::managed_shared_memory mainMemory{
            boost::interprocess::open_only,
            ourName.c_str()
    };

    boost::interprocess::managed_shared_memory ourMemory{boost::interprocess::open_only,ourName.c_str() };

    boost::interprocess::interprocess_condition *interprocess_condition =
            ourMemory.find<boost::interprocess::interprocess_condition>("barrier").first;


    boost::interprocess::managed_shared_memory partnerMemory{boost::interprocess::open_only, ourName.c_str()};



    boost::interprocess::interprocess_mutex *mtx =
            mainMemory.find_or_construct<boost::interprocess::interprocess_mutex>("mtx")();
    boost::interprocess::scoped_lock<boost::interprocess::interprocess_mutex> lock{*mtx};
    interprocess_condition->notify_all();
    std::cout << chainIndex << " is waiting" << std::endl;
    interprocess_condition->wait(lock);
    std::cout << chainIndex << " waited" << std::endl;
    interprocess_condition->notify_all();

    std::cout << chainIndex << " proposing" << std::endl;
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

std::string hops::ParallelTemperingSEOBoostInterprocess::getSharedMemoryName(int index) const {
    return std::string(sharedMemoryNameSpace) + std::to_string(index);
}


