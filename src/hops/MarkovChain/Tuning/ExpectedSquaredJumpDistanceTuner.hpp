#ifndef NEW_HOPS_EXPECTEDSQUAREDJUMPDISTANCETUNER_HPP
#define NEW_HOPS_EXPECTEDSQUAREDJUMPDISTANCETUNER_HPP

#include "hops/FileWriter/FileWriter.hpp"
#include "hops/FileWriter/FileWriterFactory.hpp"
#include "hops/FileWriter/FileWriterType.hpp"
#include "hops/MarkovChain/MarkovChain.hpp"
#include "hops/MarkovChain/Tuning/ExpectedSquaredJumpDistanceTarget.hpp"
#include "hops/MarkovChain/Tuning/ThompsonSamplingTuner.hpp"
#include "hops/Optimization/GaussianProcess.hpp"
#include "hops/Optimization/ThompsonSampling.hpp"
#include "hops/RandomNumberGenerator/RandomNumberGenerator.hpp"
#include "hops/Statistics/ExpectedSquaredJumpDistance.hpp"
#include "hops/Utility/MatrixType.hpp"
#include "hops/Utility/VectorType.hpp"

#include <Eigen/Core>

#include <chrono>
#include <cmath>
#include <memory>
#include <stdexcept>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif 

namespace hops {
    class ExpectedSquaredJumpDistanceTuner {
    public:
        struct param_type {
            ThompsonSamplingTuner::param_type ts_params;
            size_t iterationsToTestStepSize;
            std::vector<unsigned long> lags;
            bool considerTimeCost;
            bool estimateCovariance;

            param_type(size_t iterationsToTestStepSize,
                       size_t posteriorUpdateIterations,
                       size_t pureSamplingIterations,
                       size_t iterationsForConvergence,
                       size_t stepSizeGridSize,
                       double stepSizeLowerBound,
                       double stepSizeUpperBound,
                       double smoothingLength,
                       size_t randomSeed,
                       bool recordData = false,
                       std::vector<unsigned long> lags = {1},
                       bool considerTimeCost = false,
                       bool estimateCovariance = true
            );
        };

        /**
         * @brief tunes markov chain acceptance rate by nested intervals. The chain is not guaranteed to have converged
         *        to the specified acceptance rate.
         * @details Clears Markov chain history.
         * @param markovChains
         * @param parameters
         * @return true if markov chain is tuned
         */
        static bool
        tune(std::vector<std::shared_ptr<MarkovChain>>&, 
             const std::vector<RandomNumberGenerator*>&, 
             param_type&);

        /**
         * @brief tunes markov chain acceptance rate by nested intervals. The chain is not guaranteed to have converged
         *        to the specified acceptance rate.
         * @details Clears Markov chain history.
         * @param markovChains
         * @param parameters
         * @return true if markov chain is tuned
         */
        static bool
        tune(VectorType&, 
             double&,
             std::vector<std::shared_ptr<MarkovChain>>&, 
             const std::vector<RandomNumberGenerator*>&, 
             param_type&);

        /**
         * @brief tunes markov chain acceptance rate by nested intervals. The chain is not guaranteed to have converged
         *        to the specified acceptance rate.
         * @details Clears Markov chain history.
         * @param markovChains
         * @param parameters
         * @return true if markov chain is tuned
         */
        static bool
        tune(VectorType&, 
             double&,
             std::vector<std::shared_ptr<MarkovChain>>&, 
             const std::vector<RandomNumberGenerator*>&, 
             param_type&,
             Eigen::MatrixXd& data);

        ExpectedSquaredJumpDistanceTuner() = delete;
    };
}

#endif //HOPS_EXPECTEDSQUAREDJUMPDISTANCETUNER_HPP
