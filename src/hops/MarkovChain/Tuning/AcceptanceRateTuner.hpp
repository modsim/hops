#ifndef HOPS_ACCEPTANCERATETUNER_HPP
#define HOPS_ACCEPTANCERATETUNER_HPP

#include <Eigen/Core>
#include <chrono>
#include <cmath>
#include <memory>
#include <numeric>

#ifdef _OPENMP

#include <omp.h>

#endif

#include <vector>

#include "hops/FileWriter/FileWriter.hpp"
#include "hops/FileWriter/FileWriterFactory.hpp"
#include "hops/FileWriter/FileWriterType.hpp"
#include "hops/MarkovChain/MarkovChain.hpp"
#include "hops/MarkovChain/Tuning/AcceptanceRateTarget.hpp"
#include "hops/MarkovChain/Tuning/ThompsonSamplingTuner.hpp"
#include "hops/Optimization/GaussianProcess.hpp"
#include "hops/Optimization/ThompsonSampling.hpp"
#include "hops/RandomNumberGenerator/RandomNumberGenerator.hpp"


namespace hops {
    class AcceptanceRateTuner {
    public:
        struct param_type {
            ThompsonSamplingTuner::param_type ts_params;
            double acceptanceRateTargetValue;
            size_t iterationsToTestStepSize;

            param_type(double acceptanceRateTargetValue,
                       size_t iterationsToTestStepSize,
                       size_t posteriorUpdateIterations,
                       size_t pureSamplingIterations,
                       size_t iterationsForConvergence,
                       size_t stepSizeGridSize,
                       double stepSizeLowerBound,
                       double stepSizeUpperBound,
                       double smoothingLength,
                       size_t randomSeed,
                       bool recordData = false
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
        tune(std::vector<std::shared_ptr<MarkovChain>> &markovChain,
             const std::vector<RandomNumberGenerator*> &randomNumberGenerator,
             param_type &parameters);

        /**
         * @brief tunes markov chain acceptance rate by nested intervals. The chain is not guaranteed to have converged
         *        to the specified acceptance rate.
         * @details Clears Markov chain history.
         * @param markovChain
         * @param parameters
         * @return true if markov chain is tuned
         */
        static bool
        tune(VectorType &stepSize,
             double &deltaAcceptanceRate,
             std::vector<std::shared_ptr<MarkovChain>> &markovChain,
             const std::vector<RandomNumberGenerator*> &randomNumberGenerator,
             param_type &parameters);

        /**
         * @brief tunes markov chain acceptance rate by nested intervals. The chain is not guaranteed to have converged
         *        to the specified acceptance rate.
         * @details Clears Markov chain history.
         * @param markovChain
         * @param parameters
         * @return true if markov chain is tuned
         */
        static bool
        tune(VectorType &stepSize,
             double &deltaAcceptanceRate,
             std::vector<std::shared_ptr<MarkovChain>> &markovChain,
             const std::vector<RandomNumberGenerator*> &randomNumberGenerator,
             param_type &,
             Eigen::MatrixXd &);

        AcceptanceRateTuner() = delete;
    };
}

#endif //HOPS_ACCEPTANCERATETUNER_HPP
