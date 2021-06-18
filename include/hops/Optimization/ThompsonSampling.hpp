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
        using ThompsonSamplgTargetType = internal::ThompsonSamplingTarget<std::vector<typename MatrixType::Scalar>, VectorType>;

        static bool optimize (size_t numberOfRounds,
                              GaussianProcessType& initialGP,
                              std::shared_ptr<ThompsonSamplgTargetType> targetFunction,
                              const std::vector<VectorType>& parameterSpaceGrid,
                              RandomNumberGenerator randomNumberGenerator,
                              std::vector<VectorType>& samples,
                              std::vector<double>& observations,
                              std::vector<double>& noise,
                              double* rescaling = nullptr) {
            double maximumObservation = std::numeric_limits<double>::min();
            size_t maxElementIndex;
            GaussianProcess gp = initialGP.getPriorCopy();

            for (size_t i = 0; i < numberOfRounds; ++i) {
                // sample the acquisition function and obtain its maximum
                gp.sample(parameterSpaceGrid, randomNumberGenerator, maxElementIndex);
                Eigen::VectorXd testParameter = parameterSpaceGrid[maxElementIndex];

                // evaluate stepsize which maximized the sampled acquisition function
                auto[evaluations, _noise] = (*targetFunction)(testParameter);

                for (size_t j = 0; j < evaluations.size(); ++j) {
                    samples.push_back(testParameter);
                    observations.push_back(evaluations[j]);
                    noise.push_back(_noise[j]);

                    // update max observed evaluation
                    if (rescaling && evaluations[j] > maximumObservation) {
                        maximumObservation = evaluations[j];
                    }
                }

                if (rescaling) {
                    double unscaleFactor = maximumObservation;
                    // the observations which were already recorded have to be rescaled
                    for (size_t j = 0; j < observations.size() - evaluations.size(); ++j) {
                        // if the unscaleFactor is zero, then because all previous observations
                        // were zero, so they may as well be multiplied with zero
                        observations[j] *= unscaleFactor;  
                        // if the new maximum is zero, then also the old was, so no rescaling is done (=division by one)
                        observations[j] /= (maximumObservation != 0 ? maximumObservation : 1);
                    }

                    // the new observations have not yet been scaled
                    for (size_t j = observations.size() - evaluations.size(); j < observations.size(); ++j) {
                        // if the new maximum is zero, then also the old was, so no rescaling is done (=division by one)
                        observations[j] /= (maximumObservation != 0 ? maximumObservation : 1);
                    }

                    gp = initialGP.getPriorCopy();
                    gp.addObservations(samples, observations, noise);
                } else {
                    gp.addObservations(std::vector<Eigen::VectorXd>(evaluations.size(), testParameter), 
                                       evaluations, 
                                       _noise);
                }
            }

            gp.sample(parameterSpaceGrid, randomNumberGenerator, maxElementIndex);
            initialGP = gp.getPosteriorCopy();

            if (rescaling) {
                *rescaling = maximumObservation;
            }

            return true;
        }

        static bool optimize (size_t numberOfRounds,
                              GaussianProcessType gp,
                              std::shared_ptr<ThompsonSamplgTargetType> targetFunction,
                              const std::vector<VectorType>& parameterSpaceGrid,
                              RandomNumberGenerator randomNumberGenerator,
                              double noise = 0, 
                              bool rescaling = false) {
            std::vector<Eigen::VectorXd> samples;
            std::vector<double> observations;
            
            return optimize(numberOfRounds, 
                            targetFunction, 
                            parameterSpaceGrid, 
                            samples, 
                            observations, 
                            noise, 
                            rescaling);
        }
    };
}

#endif // HOPS_THOMPSONSAMPLING_HPP
