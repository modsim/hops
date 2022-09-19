#ifndef HOPS_THOMPSONSAMPLINGTUNER_HPP
#define HOPS_THOMPSONSAMPLINGTUNER_HPP

#include <Eigen/Core>
#include <chrono>
#include <cmath>
#include <memory>
#include <stdexcept>

#include "hops/FileWriter/FileWriter.hpp"
#include "hops/FileWriter/FileWriterFactory.hpp"
#include "hops/FileWriter/FileWriterType.hpp"
#include "hops/MarkovChain/MarkovChain.hpp"
#include "hops/Optimization/GaussianProcess.hpp"
#include "hops/Optimization/ThompsonSampling.hpp"
#include "hops/Statistics/ExpectedSquaredJumpDistance.hpp"
#include "hops/Utility/MatrixType.hpp"
#include "hops/Utility/VectorType.hpp"

namespace hops {
    class ThompsonSamplingTuner {
    public:
        struct param_type {
            size_t posteriorUpdateIterations;
            size_t pureSamplingIterations;
            size_t iterationsForConvergence;
            size_t posteriorUpdateIterationsNeeded;
            size_t stepSizeGridSize;
            double stepSizeLowerBound;
            double stepSizeUpperBound;
            double smoothingLength;
            size_t randomSeed;
            bool recordData;

            param_type() = default;

            param_type(size_t posteriorUpdateIterations,
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
        template<typename TuningTargetType>
        static bool
        tune(const std::vector<RandomNumberGenerator*>&, 
             param_type&,
             TuningTargetType&);

        /**
         * @brief tunes markov chain acceptance rate by nested intervals. The chain is not guaranteed to have converged
         *        to the specified acceptance rate.
         * @details Clears Markov chain history.
         * @param markovChain
         * @param parameters
         * @return true if markov chain is tuned
         */
        template<typename TuningTargetType>
        static bool
        tune(VectorType&, 
             double&,
             const std::vector<RandomNumberGenerator*>&, 
             param_type&,
             TuningTargetType&);
        /**
         * @brief tunes markov chain acceptance rate by nested intervals. The chain is not guaranteed to have converged
         *        to the specified acceptance rate.
         * @details Clears Markov chain history.
         * @param markovChain
         * @param parameters
         * @return true if markov chain is tuned
         */
        template<typename TuningTargetType>
        static bool
        tune(VectorType&, 
             double&,
             const std::vector<RandomNumberGenerator*>&, 
             param_type&,
             TuningTargetType&,
             MatrixType& data);

        ThompsonSamplingTuner() = delete;
    };
}

template<typename TuningTargetType>
bool hops::ThompsonSamplingTuner::tune(
        VectorType& stepSize,
        double& maximumTargetValue,
        const std::vector<RandomNumberGenerator*>& targetRandomNumberGenerators,
        hops::ThompsonSamplingTuner::param_type& parameters,
        TuningTargetType& target,
        MatrixType& data) {
    using Kernel = SquaredExponentialKernel<MatrixType, VectorType>;
    using GP = GaussianProcess<MatrixType, VectorType, Kernel>;

    VectorType logStepSizeGrid(parameters.stepSizeGridSize);
    double a = std::log10(parameters.stepSizeLowerBound), b = std::log10(parameters.stepSizeUpperBound);

    const auto indexToValue = [=] (size_t i) -> double { return (b - a) * i / (parameters.stepSizeGridSize - 1) + a; };
    const auto valueToIndex = [=] (double v) -> size_t { return std::round((parameters.stepSizeGridSize - 1) * (v - a) / (b - a)); };

    for (size_t i = 0; i < parameters.stepSizeGridSize; ++i) {
        logStepSizeGrid(i) = indexToValue(i);
        assert(i == valueToIndex(logStepSizeGrid(i)));
    }

    double sigma = 1, length = 1;
    Kernel kernel(sigma, length);
    GP gp = GP(kernel);

    RandomNumberGenerator thompsonSamplingRandomNumberGenerator(parameters.randomSeed, targetRandomNumberGenerators.size());
    bool isThompsonSamplingConverged = ThompsonSampling<GP, TuningTargetType>::optimize(
            parameters.posteriorUpdateIterations,
            parameters.pureSamplingIterations,
            parameters.iterationsForConvergence,
            gp, target, logStepSizeGrid, 
            targetRandomNumberGenerators,
            thompsonSamplingRandomNumberGenerator,
            &parameters.posteriorUpdateIterationsNeeded,
            parameters.smoothingLength);
   
    if (parameters.recordData) {
        auto& posteriorMean = gp.getPosteriorMean();
        auto& posteriorCovariance = gp.getPosteriorCovariance();

        auto& observedInputs = gp.getObservedInputs();
        auto& observedValues = gp.getObservedValues();
        auto& observedValueErrors = gp.getObservedValueErrors();

        // only for logging purposes
        data = MatrixType::Zero(posteriorMean.size(), 6);
        for (long i = 0; i < posteriorMean.size(); ++i) {
            data(i, 0) = logStepSizeGrid(i, 0);
            data(i, 1) = posteriorMean(i);
            data(i, 2) = posteriorCovariance(i,i);
        }

        for (long i = 0; i < observedInputs.size(); ++i) {
            long j = valueToIndex(observedInputs(i, 0)); // observed inputs can be a vector in general, it is a scalar for stepsize-only tuning
            data(j, 3) = observedInputs(i, 0);
            data(j, 4) = observedValues(i);
            data(j, 5) = observedValueErrors(i);
        }
    }

    // store results in reference parameters
    auto& posteriorMean = gp.getPosteriorMean();
    size_t maximumIndex;
    maximumTargetValue = posteriorMean.maxCoeff(&maximumIndex);
    stepSize = VectorType::Ones(1) * std::pow(10, logStepSizeGrid(maximumIndex, 0));

    return isThompsonSamplingConverged;
}

template<typename TuningTargetType>
bool hops::ThompsonSamplingTuner::tune(const std::vector<RandomNumberGenerator*>& randomNumberGenerator,
                                       hops::ThompsonSamplingTuner::param_type& parameters,
        TuningTargetType& target) {
    VectorType stepSize;
    double maximumTargetValue;
    return tune(stepSize, maximumTargetValue, randomNumberGenerator, parameters, target);
}

template<typename TuningTargetType>
bool hops::ThompsonSamplingTuner::tune(VectorType& stepSize,
                                       double& maximumTargetValue,
                                       const std::vector<RandomNumberGenerator*>& randomNumberGenerator,
                                       hops::ThompsonSamplingTuner::param_type& parameters,
                                       TuningTargetType& target) {
    MatrixType data;
    return tune(stepSize, maximumTargetValue, randomNumberGenerator, parameters, target, data);
}

#endif // HOPS_THOMPSONSAMPLINGTUNER_HPP
