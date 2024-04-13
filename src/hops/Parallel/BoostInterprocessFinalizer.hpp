#ifndef HOPS_BOOSTINTERPROCESSFINALIZER_HPP
#define HOPS_BOOSTINTERPROCESSFINALIZER_HPP

#include <boost/interprocess/shared_memory_object.hpp>
#include <boost/interprocess/mapped_region.hpp>
#include <string>

namespace hops {
    class BoostInterprocessFinalizer {
    public:
        inline static const char *const shared_memory_name = "HOPS_shared_memory";

        BoostInterprocessFinalizer() = delete;

        static void initializeAndQueueFinalizeAtExit() {
            boost::interprocess::shared_memory_object shm_obj(
                    boost::interprocess::open_or_create,
                    shared_memory_name,
                    boost::interprocess::read_write);
            std::atexit(finalize);
        }

        static void finalize() {
            std::cout << "call finalize" << std::endl;
            std::cout << boost::interprocess::shared_memory_object::remove(shared_memory_name);
        }

    };

}

#endif //HOPS_BOOSTINTERPROCESSFINALIZER_HPP
