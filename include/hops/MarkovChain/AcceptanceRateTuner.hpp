#ifndef HOPS_ACCEPTANCERATETUNER_HPP
#define HOPS_ACCEPTANCERATETUNER_HPP

#include "MarkovChain.hpp"

namespace hops {
    class AcceptanceRateTuner {
    public:
        struct param_type {
            double lowerLimitAcceptanceRate{};
            double upperLimitAcceptanceRate{};
            mutable double lowerLimitStepSize;
            mutable double upperLimitStepSize;
            size_t iterationsToTestStepSize;
            size_t maximumTotalIterations;

            param_type(double lowerLimitAcceptanceRate,
                       double upperLimitAcceptanceRate,
                       double lowerLimitStepSize,
                       double upperLimitStepSize,
                       size_t iterationsToTestStepSize,
                       size_t maximumTotalIterations
            );
        };

        /**
         * @brief tunes markov chain acceptance rate by nested intervals. The chain is not guaranteed to have converged
         *        to the specified acceptance rate.
         * @details Clears Markov chain history.
         * @param markovChain
         * @param parameters
         * @return true if markov chain is tuned
         */
        static bool
        tune(MarkovChain *markovChain, RandomNumberGenerator &randomNumberGenerator, const param_type &parameters);

        AcceptanceRateTuner() = delete;
    };
}


#endif //HOPS_ACCEPTANCERATETUNER_HPP
