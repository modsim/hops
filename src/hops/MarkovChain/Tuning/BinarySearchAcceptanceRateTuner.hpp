#ifndef HOPS_BINARYSEARCHACCEPTANCERATETUNER_HPP
#define HOPS_BINARYSEARCHACCEPTANCERATETUNER_HPP

#include "hops/MarkovChain/MarkovChain.hpp"

namespace hops {

    /**
     * @brief Deprecated as there are issues due to the uncertainty in estimating acceptance rates.
     * @deprecated Binary search does not work well for acceptance rate tuning, because the acceptance rate is uncertain.
     */
    class BinarySearchAcceptanceRateTuner {
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
         * size_t indiciation number of iterations used and the tuned MarkovChain
         * @return true if markov chain is tuned
         */
        static bool
        tune(MarkovChain *markovChain, 
             RandomNumberGenerator &randomNumberGenerator, 
             const param_type &parameters);

        /**
         * @brief tunes markov chain acceptance rate by nested intervals. The chain is not guaranteed to have converged
         *        to the specified acceptance rate.
         * @details Clears Markov chain history.
         * @param markovChain
         * @param parameters
         * @return true if markov chain is tuned
         */
        static bool
        tune(double& stepSize, 
             double& AcceptanceRate, 
             MarkovChain *markovChain, 
             RandomNumberGenerator &randomNumberGenerator, 
             const param_type &parameters);

        /**
         * @brief tunes markov chain acceptance rate by nested intervals. The chain is not guaranteed to have converged
         *        to the specified acceptance rate.
         * @details Clears Markov chain history.
         * @param markovChain
         * @param parameters
         * @return true if markov chain is tuned
         */
        static bool
        tune(std::vector<std::shared_ptr<MarkovChain>>& markovChain, 
             std::vector<RandomNumberGenerator>& randomNumberGenerator, 
             const param_type &parameters);

        /**
         * @brief tunes markov chain acceptance rate by nested intervals. The chain is not guaranteed to have converged
         *        to the specified acceptance rate.
         * @details Clears Markov chain history.
         * @param markovChain
         * @param parameters
         * @return true if markov chain is tuned
         */
        static bool
        tune(double& stepSize, 
             double& acceptanceRate,
             std::vector<std::shared_ptr<MarkovChain>>& markovChain, 
             std::vector<RandomNumberGenerator>& randomNumberGenerator, 
             const param_type &parameters);

        BinarySearchAcceptanceRateTuner() = delete;
    };
}

#endif //HOPS_BINARYSEARCHACCEPTANCERATETUNER_HPP
