#ifndef NEW_HOPS_ACCEPTANCERATETUNER_HPP
#define NEW_HOPS_ACCEPTANCERATETUNER_HPP

#include <hops/FileWriter/FileWriter.hpp>
#include <hops/FileWriter/FileWriterFactory.hpp>
#include <hops/FileWriter/FileWriterType.hpp>
#include <hops/MarkovChain/MarkovChain.hpp>
#include <hops/MarkovChain/MarkovChainAttribute.hpp>
#include <hops/MarkovChain/Tuning/AcceptanceRateTuner.hpp>
#include <hops/Optimization/GaussianProcess.hpp>
#include <hops/Optimization/ThompsonSampling.hpp>

#include <Eigen/Core>

#include <chrono>
#include <cmath>
#include <memory>
#include <vector>

namespace hops {
    class AcceptanceRateTuner {
    public:
        struct param_type {
            double acceptanceRateTargetValue;
            size_t iterationsToTestStepSize;
            size_t posteriorUpdateIterations;
            size_t pureSamplingIterations;
            size_t iterationsForConvergence;
            size_t posteriorUpdateIterationsNeeded;
            size_t stepSizeGridSize;
            double stepSizeLowerBound;
            double stepSizeUpperBound;
            size_t randomSeed;
            std::string outputDirectory;

            param_type(double acceptanceRateTargetValue,
                       size_t iterationsToTestStepSize,
                       size_t posteriorUpdateIterations,
                       size_t pureSamplingIterations,
                       size_t iterationsForConvergence,
                       size_t stepSizeGridSize,
                       double stepSizeLowerBound,
                       double stepSizeUpperBound,
                       size_t randomSeed,
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
        tune(std::vector<std::shared_ptr<MarkovChain>>& markovChain, 
             std::vector<RandomNumberGenerator>& randomNumberGenerator, 
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
        tune(double& stepSize, 
             double& deltaAcceptanceRate, 
             std::vector<std::shared_ptr<MarkovChain>>& markovChain, 
             std::vector<RandomNumberGenerator>& randomNumberGenerator, 
             param_type &parameters);

        AcceptanceRateTuner() = delete;
    };

    namespace internal {
        struct AcceptanceRateTarget : public ThompsonSamplingTarget<double, Eigen::VectorXd> {
            std::vector<std::shared_ptr<hops::MarkovChain>> markovChain;
            std::vector<RandomNumberGenerator>* randomNumberGenerator;
            AcceptanceRateTuner::param_type parameters;

            AcceptanceRateTarget(std::vector<std::shared_ptr<hops::MarkovChain>>& markovChain,
                                              std::vector<hops::RandomNumberGenerator>& randomNumberGenerator,
                                              const hops::AcceptanceRateTuner::param_type& parameters) :
                    markovChain(markovChain),
                    randomNumberGenerator(&randomNumberGenerator),
                    parameters(parameters) {}

            virtual std::tuple<double, double> operator()(const Eigen::VectorXd& x) override;
        };
    }
}

#endif //HOPS_ACCEPTANCERATETUNER_HPP
