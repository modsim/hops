#ifndef HOPS_THOMPSONSAMPLING_HPP
#define HOPS_THOMPSONSAMPLING_HPP

#include <hops/Optimization/GaussianProcess.hpp>

#include <cmath>
#include <limits>
#include <memory>
#include <chrono>

namespace hops {
    namespace internal {
        template<typename ReturnType, typename InputType>
        struct ThompsonSamplingTarget {
            virtual std::tuple<ReturnType, ReturnType> operator()(const InputType&) = 0;
        };
    }

    template<typename MatrixType, typename VectorType, typename GaussianProcessType>
    class ThompsonSampling {
    public:
        using ThompsonSamplgTargetType = internal::ThompsonSamplingTarget<typename MatrixType::Scalar, VectorType>;

        static bool optimize (const size_t numberOfPosteriorUpdates,
                              const size_t numberOfSamplingRounds,
                              const size_t numberOfRoundsForConvergence,
                              GaussianProcessType& initialGP,
                              std::shared_ptr<ThompsonSamplgTargetType> targetFunction,
                              const std::vector<VectorType>& inputSpaceGrid,
                              RandomNumberGenerator randomNumberGenerator,
                              size_t* numberOfPosteriorUpdatesNeeded = nullptr) {
            size_t maxElementIndex;
            bool isConverged = false;

            size_t newMaximumPosteriorMeanIndex = 0, oldMaximumPosteriorMeanIndex = 0, sameMaximumCounter = 0;
            GaussianProcess gp = initialGP.getPriorCopy();

            for (size_t i = 0; i < numberOfPosteriorUpdates; ++i) {
                std::unordered_map<size_t, size_t> observedInputIndex;
                std::vector<size_t> observedInputCount;
                std::vector<Eigen::VectorXd> observedInput;
                std::vector<double> observedValueMean;
                std::vector<double> observedValueErrorMean;

                for (size_t j = 0; j < numberOfSamplingRounds; ++j) {
                    // sample the acquisition function and obtain its maximum
                    gp.sample(inputSpaceGrid, randomNumberGenerator, maxElementIndex);
                    Eigen::VectorXd testInput = inputSpaceGrid[maxElementIndex];

                    // evaluate stepsize which maximized the sampled acquisition function
                    auto[newObservedValue, newObservedValueError] = (*targetFunction)(testInput);

                    // aggregate data if this stepsize has been tested before
                    if (observedInputIndex.count(maxElementIndex)) {
                        size_t k = observedInputIndex[maxElementIndex];
                        size_t m = observedInputCount[k];
                        observedInputCount[k] += 1;
                        double oldObservedValueMean = observedValueMean[k]; // store for variance update
                        observedValueMean[k] = (m * observedValueMean[k] + newObservedValue) / (m+1);
                        observedValueErrorMean[k] = 
                                (m * (std::pow(observedValueErrorMean[k], 2) + std::pow(oldObservedValueMean, 2)) + 
                                 1 * (std::pow(newObservedValueError, 2) + std::pow(newObservedValue, 2))) / (m+1) - std::pow(observedValueMean[k], 2);
                    } else {
                        observedInputIndex[maxElementIndex] = observedInput.size();
                        observedInputCount.push_back(1);
                        observedInput.push_back(testInput);
                        observedValueMean.push_back(newObservedValue);
                        observedValueErrorMean.push_back(newObservedValueError);
                    }
                }

                gp.updateObservations(observedInput, observedValueMean, observedValueErrorMean);
                gp.addObservations(observedInput, observedValueMean, observedValueErrorMean);

                // check maximum of posterior mean and increment counter, if the index didnt change
                // or reset counter, if we have a new maximum
                gp.getPosteriorMean().maxCoeff(&newMaximumPosteriorMeanIndex);
                if (newMaximumPosteriorMeanIndex != oldMaximumPosteriorMeanIndex) {
                    sameMaximumCounter = 0;
                } else {
                    ++sameMaximumCounter;
                }
                oldMaximumPosteriorMeanIndex = newMaximumPosteriorMeanIndex;

                // if the posterior mean hasn't change for the last sameMaximumCounter of rounds, 
                // then we assume convergence and break
                if (sameMaximumCounter == numberOfRoundsForConvergence) {
                    isConverged = true;
                    if (numberOfPosteriorUpdatesNeeded) {
                        *numberOfPosteriorUpdatesNeeded = i;
                    }
                    break;
                }
            }

            gp.sample(inputSpaceGrid, randomNumberGenerator, maxElementIndex);
            initialGP = gp.getPosteriorCopy();

            return isConverged;
        }
    };
}

#endif // HOPS_THOMPSONSAMPLING_HPP
