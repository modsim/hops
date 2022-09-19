#ifndef HOPS_SIMPLEEXPECTEDSQUAREDJUMPDISTANCETUNER_HPP
#define HOPS_SIMPLEEXPECTEDSQUAREDJUMPDISTANCETUNER_HPP

#include "hops/Statistics/ExpectedSquaredJumpDistance.hpp"
#include "hops/FileWriter/FileWriter.hpp"
#include "hops/FileWriter/FileWriterFactory.hpp"
#include "hops/FileWriter/FileWriterType.hpp"
#include "hops/MarkovChain/MarkovChain.hpp"

#include <chrono>
#include <cmath>
#include <memory>
#include <stdexcept>
#include <vector>

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
            std::string outputDirectory;

            param_type(size_t iterationsToTestStepSize,
                       //size_t maximumTotalIterations,
                       size_t stepSizeGridSize,
                       double stepSizeLowerBound,
                       double stepSizeUpperBound,
                       bool considerTimeCost,
                       std::string outputDirectory
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
