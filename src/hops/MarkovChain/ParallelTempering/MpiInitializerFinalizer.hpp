#ifndef HOPS_MPIINITIALIZERFINALIZER_HPP
#define HOPS_MPIINITIALIZERFINALIZER_HPP

#include <mpi.h>

namespace hops {
    class MpiInitializerFinalizer {
    public:
        MpiInitializerFinalizer() = delete;

        static void initializeAndQueueFinalizeAtExit() {
            int isMpiInitialized;
            MPI_Initialized(&isMpiInitialized);
            if (!isMpiInitialized) {
                MPI_Init(NULL, NULL);
                std::atexit(finalize);
            }
        }

        static void finalize() {
            int isMpiFinalized;
            MPI_Finalized(&isMpiFinalized);
            if (!isMpiFinalized) {
                int isFinalizeSuccessful = !MPI_Finalize();
                if (!isFinalizeSuccessful) {
                    throw std::runtime_error("MPI failed to finalize.");
                }
            }
        }

        static const int &getInternalMpiTag() {
            return INTERNAL_MPI_TAG;
        }

    private:
        constexpr static int INTERNAL_MPI_TAG = 137;
    };
}

#endif //HOPS_MPIINITIALIZERFINALIZER_HPP
