#ifndef HOPS_SIMPLEEXPECTEDSQUAREDJUMPDISTANCETUNER_HPP
#define HOPS_SIMPLEEXPECTEDSQUAREDJUMPDISTANCETUNER_HPP

#include "MarkovChain.hpp"
#include "../Diagnostics/ExpectedSquaredJumpDistance.hpp"
#include <memory>
#include <stdexcept>

namespace hops {
    class SimpleExpectedSquaredJumpDistanceTuner {
    public:
        struct param_type {
            size_t iterationsToTestStepSize;
            //size_t maximumTotalIterations;
            size_t stepSizeGridSize;
            double stepSizeLowerBound;
            double stepSizeUpperBound;
            bool considerTimeCost;

            param_type(size_t iterationsToTestStepSize,
                       //size_t maximumTotalIterations,
                       size_t stepSizeGridSize,
                       double stepSizeLowerBound,
                       double stepSizeUpperBound,
                       bool considerTimeCost
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
        tune(std::vector<std::shared_ptr<MarkovChain>>&, 
             std::vector<RandomNumberGenerator>&, 
             const param_type&);

        /**
         * @brief tunes markov chain acceptance rate by nested intervals. The chain is not guaranteed to have converged
         *        to the specified acceptance rate.
         * @details Clears Markov chain history.
         * @param markovChain
         * @param parameters
         * @return true if markov chain is tuned
         */
        static bool
        tune(double&, 
             double&,
             std::vector<std::shared_ptr<MarkovChain>>&, 
             std::vector<RandomNumberGenerator>&, 
             const param_type&);

        SimpleExpectedSquaredJumpDistanceTuner() = delete;
    };
}

#endif //HOPS_SIMPLEEXPECTEDSQUAREDJUMPDISTANCETUNER_HPP
