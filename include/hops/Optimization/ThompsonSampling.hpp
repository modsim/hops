#ifndef HOPS_THOMPSONSAMPLING_HPP
#define HOPS_THOMPSONSAMPLING_HPP

#include <hops/Optimization/GaussianProcess.hpp>
#include <hops/Optimization/Kernel/UniformBallKernel.hpp>
#include <hops/Optimization/Kernel/SquaredExponentialKernel.hpp>
#include <hops/Optimization/Kernel/ZeroKernel.hpp>

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
                              const MatrixType& inputSpaceGrid,
                              RandomNumberGenerator randomNumberGenerator,
                              size_t* numberOfPosteriorUpdatesNeeded = nullptr,
                              double smoothingLength = 0) {
            size_t maxElementIndex;
            bool isConverged = false;

            size_t newMaximumPosteriorMeanIndex = 0, oldMaximumPosteriorMeanIndex = 0, sameMaximumCounter = 0;
            GaussianProcess gp = initialGP.getPriorCopy();

            UniformBallKernel smoothingKernel = UniformBallKernel<MatrixType, VectorType>(smoothingLength);
            GaussianProcess data = GaussianProcess<MatrixType, VectorType, decltype(smoothingKernel)>(smoothingKernel);

            size_t i = 0;
            for (; i < numberOfPosteriorUpdates; ++i) {
                // if the posterior mean hasn't change for the last sameMaximumCounter of rounds, 
                // then we assume convergence and break
                if (sameMaximumCounter == numberOfRoundsForConvergence) {
                    isConverged = true;
                    break;
                }

                std::unordered_map<long, long> observedInputIndex;
                std::vector<long> observedInputCount;
                MatrixType observedInput(0, inputSpaceGrid.cols());
                VectorType observedValueMean(0);
                VectorType observedValueErrorMean(0);

                for (size_t j = 0; j < numberOfSamplingRounds; ++j) {
                    // sample the acquisition function and obtain its maximum
                    gp.sample(inputSpaceGrid, randomNumberGenerator, maxElementIndex);
                    VectorType testInput = inputSpaceGrid.row(maxElementIndex);

                    // evaluate stepsize which maximized the sampled acquisition function
                    auto[newObservedValue, newObservedValueError] = (*targetFunction)(testInput);

                    // aggregate data if this stepsize has been tested before
                    if (observedInputIndex.count(maxElementIndex)) {
                        size_t k = observedInputIndex[maxElementIndex];

                        // increment counter
                        size_t m = observedInputCount[k];
                        observedInputCount[k] += 1;

                        double oldObservedValueMean = observedValueMean(k); // store for variance update
                        observedValueMean(k) = (m * observedValueMean(k) + newObservedValue) / (m+1);
                        observedValueErrorMean(k) = 
                                (m * (std::pow(observedValueErrorMean(k), 2) + std::pow(oldObservedValueMean, 2)) + 
                                 1 * (std::pow(newObservedValueError, 2) + std::pow(newObservedValue, 2))) / (m+1) - std::pow(observedValueMean(k), 2);
                    } else {
                        size_t n = observedInput.rows();
                        observedInputIndex[maxElementIndex] = n;
                        observedInputCount.push_back(1);

                        observedInput = internal::append(observedInput, testInput);
                        observedValueMean = internal::append(observedValueMean, newObservedValue);
                        observedValueErrorMean = internal::append(observedValueErrorMean, newObservedValueError);
                    }
                }

                // update the data to have the combined observed inputs, values and errors
                std::tie(observedInput, observedValueMean, observedValueErrorMean) = data.updateObservations(
                        observedInput, observedValueMean, observedValueErrorMean);
                data.addObservations(observedInput, observedValueMean, observedValueErrorMean);

                // compute smoothing on unsmoothed errors
                //MatrixType weights = smoothingKernel(data.getObservedInputs(), data.getObservedInputs());
                const MatrixType& weights = data.getObservedCovariance();
                VectorType smoothedObservedValueErrors = weights * data.getObservedValueErrors();
                //VectorType normalizer = weights * VectorType::Ones(weights.cols());
                VectorType normalizer = weights.rowwise().sum(); // * VectorType::Ones(weights.cols());
                smoothedObservedValueErrors = smoothedObservedValueErrors.array() / normalizer.array();

                //MatrixType print(smoothedObservedValueErrors.rows(), 2);
                //print << data.getObservedValueErrors(), smoothedObservedValueErrors;
                //std::cout << print << std::endl << std::endl;

                std::tie(observedInput, observedValueMean, smoothedObservedValueErrors) = gp.updateObservations(
                        data.getObservedInputs(), data.getObservedValues(), smoothedObservedValueErrors);
                gp.addObservations(observedInput, observedValueMean, smoothedObservedValueErrors);

                double newKernelSigma = 2*(data.getObservedValues().array() + data.getObservedValueErrors().array().sqrt()).maxCoeff();
                gp.setKernelSigma(newKernelSigma);

                // check maximum of posterior mean and increment counter, if the index didnt change
                // or reset counter, if we have a new maximum
                // also, as long as the maximum is zero, we keep exploring
                auto max = gp.getPosteriorMean().maxCoeff(&newMaximumPosteriorMeanIndex);
                if (newMaximumPosteriorMeanIndex != oldMaximumPosteriorMeanIndex || max == 0) {
                    sameMaximumCounter = 0;
                } else {
                    ++sameMaximumCounter;
                }
                oldMaximumPosteriorMeanIndex = newMaximumPosteriorMeanIndex;
            }

            if (numberOfPosteriorUpdatesNeeded) {
                *numberOfPosteriorUpdatesNeeded = i;
            }

            gp.sample(inputSpaceGrid, randomNumberGenerator, maxElementIndex);
            initialGP = gp.getPosteriorCopy();

            return isConverged;
        }
    };
}

#endif // HOPS_THOMPSONSAMPLING_HPP
