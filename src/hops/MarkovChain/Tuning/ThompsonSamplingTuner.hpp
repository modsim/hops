#ifndef HOPS_THOMPSONSAMPLINGTUNER_HPP
#define HOPS_THOMPSONSAMPLINGTUNER_HPP

#include <hops/FileWriter/FileWriter.hpp>
#include <hops/FileWriter/FileWriterFactory.hpp>
#include <hops/FileWriter/FileWriterType.hpp>
#include <hops/MarkovChain/MarkovChain.hpp>
#include <hops/MarkovChain/MarkovChainAttribute.hpp>
#include <hops/Optimization/GaussianProcess.hpp>
#include <hops/Optimization/ThompsonSampling.hpp>
#include <hops/Statistics/ExpectedSquaredJumpDistance.hpp>
#include <hops/Utility/MatrixType.hpp>
#include <hops/Utility/VectorType.hpp>

#include <Eigen/Core>

#include <chrono>
#include <cmath>
#include <memory>
#include <stdexcept>

namespace hops {
    class ThompsonSamplingTuner {
    public:
        struct param_type {
            size_t iterationsToTestStepSize;
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

            param_type(size_t iterationsToTestStepSize,
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
        template<typename TuningTarget>
        static bool
        tune(std::vector<std::shared_ptr<MarkovChain>>&, 
             std::vector<RandomNumberGenerator*>&, 
             param_type&,
             TuningTarget&);

        /**
         * @brief tunes markov chain acceptance rate by nested intervals. The chain is not guaranteed to have converged
         *        to the specified acceptance rate.
         * @details Clears Markov chain history.
         * @param markovChain
         * @param parameters
         * @return true if markov chain is tuned
         */
        template<typename TuningTarget>
        static bool
        tune(VectorType&, 
             double&,
             std::vector<std::shared_ptr<MarkovChain>>&, 
             std::vector<RandomNumberGenerator*>&, 
             param_type&,
             TuningTarget&);
        /**
         * @brief tunes markov chain acceptance rate by nested intervals. The chain is not guaranteed to have converged
         *        to the specified acceptance rate.
         * @details Clears Markov chain history.
         * @param markovChain
         * @param parameters
         * @return true if markov chain is tuned
         */
        template<typename TuningTarget>
        static bool
        tune(VectorType&, 
             double&,
             std::vector<std::shared_ptr<MarkovChain>>&, 
             std::vector<RandomNumberGenerator*>&, 
             param_type&,
             TuningTarget&,
             MatrixType& data,
             MatrixType& posterior);

        ThompsonSamplingTuner() = delete;
    };
}

template<typename TuningTarget>
bool hops::ThompsonSamplingTuner::tune(
        VectorType& stepSize,
        double& maximumTargetValue,
        std::vector<std::shared_ptr<hops::MarkovChain>>& markovChain,
        std::vector<RandomNumberGenerator*>& randomNumberGenerator,
        hops::ThompsonSamplingTuner::param_type& parameters,
        TuningTarget& target,
        MatrixType& data,
        MatrixType& posterior) {
    using Kernel = SquaredExponentialKernel<MatrixType, VectorType>;
    using GP = GaussianProcess<MatrixType, VectorType, Kernel>;

    VectorType logStepSizeGrid(parameters.stepSizeGridSize);
    double a = std::log10(parameters.stepSizeLowerBound), b = std::log10(parameters.stepSizeUpperBound);

    for (size_t i = 0; i < parameters.stepSizeGridSize; ++i) {
        logStepSizeGrid(i) = (b - a) * i / (parameters.stepSizeGridSize - 1) + a;
    }

    double sigma = 1, length = 1;
    Kernel kernel(sigma, length);
    GP gp = GP(kernel);

    target.markovChain = markovChain;
    target.randomNumberGenerator = randomNumberGenerator;
    target.numberOfTestSamples = parameters.iterationsToTestStepSize;

    RandomNumberGenerator thompsonSamplingRandomNumberGenerator(parameters.randomSeed, markovChain.size() + 1);
    bool isThompsonSamplingConverged = ThompsonSampling<MatrixType, VectorType, GP, TuningTarget>::optimize(
            parameters.posteriorUpdateIterations,
            parameters.pureSamplingIterations,
            parameters.iterationsForConvergence,
            gp, target, logStepSizeGrid, 
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
        posterior = MatrixType(posteriorMean.size(), 3);
        for (long i = 0; i < posteriorMean.size(); ++i) {
            posterior(i, 0) = logStepSizeGrid(i, 0);
            posterior(i, 1) = posteriorMean(i);
            posterior(i, 2) = posteriorCovariance(i,i);
        }

        // only for logging purposes
        data = MatrixType(observedInputs.size(), 3);
        for (long i = 0; i < observedInputs.size(); ++i) {
            data(i, 0) = observedInputs(i, 0);
            data(i, 1) = observedValues(i);
            data(i, 2) = observedValueErrors(i);
        }
    }

    // store results in reference parameters
    auto& posteriorMean = gp.getPosteriorMean();
    size_t maximumIndex;
    maximumTargetValue = posteriorMean.maxCoeff(&maximumIndex);
    stepSize = VectorType::Ones(1) * std::pow(10, logStepSizeGrid(maximumIndex, 0));

    return isThompsonSamplingConverged;
}

template<typename TuningTarget>
bool hops::ThompsonSamplingTuner::tune(
        std::vector<std::shared_ptr<hops::MarkovChain>>& markovChain,
        std::vector<RandomNumberGenerator*>& randomNumberGenerator,
        hops::ThompsonSamplingTuner::param_type& parameters,
        TuningTarget& target) {
    VectorType stepSize = std::any_cast<double>(markovChain[0]->getParameter(ProposalParameter::STEP_SIZE)) * VectorType::Ones(1);
    double maximumTargetValue;
    return tune(stepSize, maximumTargetValue, markovChain, randomNumberGenerator, parameters, target);
}

template<typename TuningTarget>
bool hops::ThompsonSamplingTuner::tune(
        VectorType& stepSize,
        double& maximumTargetValue,
        std::vector<std::shared_ptr<hops::MarkovChain>>& markovChain,
        std::vector<RandomNumberGenerator*>& randomNumberGenerator,
        hops::ThompsonSamplingTuner::param_type& parameters,
        TuningTarget& target) {
    MatrixType data, posterior;
    return tune(stepSize, maximumTargetValue, markovChain, randomNumberGenerator, parameters, target, data, posterior);
}

hops::ThompsonSamplingTuner::param_type::param_type(size_t iterationsToTestStepSize,
                                                               size_t posteriorUpdateIterations,
                                                               size_t pureSamplingIterations,
                                                               size_t iterationsForConvergence,
                                                               size_t stepSizeGridSize,
                                                               double stepSizeLowerBound,
                                                               double stepSizeUpperBound,
                                                               double smoothingLength,
                                                               size_t randomSeed,
                                                               bool recordData) {
    this->iterationsToTestStepSize = iterationsToTestStepSize;
    this->posteriorUpdateIterations = posteriorUpdateIterations;
    this->pureSamplingIterations = pureSamplingIterations;
    this->iterationsForConvergence = iterationsForConvergence;
    this->posteriorUpdateIterationsNeeded = 0;
    this->stepSizeGridSize = stepSizeGridSize;
    this->stepSizeLowerBound = stepSizeLowerBound;
    this->stepSizeUpperBound = stepSizeUpperBound;
    this->smoothingLength = smoothingLength;
    this->randomSeed = randomSeed;
    this->recordData = recordData;
}

#endif // HOPS_THOMPSONSAMPLINGTUNER_HPP
