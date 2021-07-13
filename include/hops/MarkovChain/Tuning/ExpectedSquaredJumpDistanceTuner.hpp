#ifndef NEW_HOPS_EXPECTEDSQUAREDJUMPDISTANCETUNER_HPP
#define NEW_HOPS_EXPECTEDSQUAREDJUMPDISTANCETUNER_HPP

#include <hops/Statistics/ExpectedSquaredJumpDistance.hpp>
#include <hops/FileWriter/FileWriter.hpp>
#include <hops/FileWriter/FileWriterFactory.hpp>
#include <hops/FileWriter/FileWriterType.hpp>
#include <hops/MarkovChain/MarkovChain.hpp>
#include <hops/MarkovChain/MarkovChainAttribute.hpp>
#include <hops/Optimization/GaussianProcess.hpp>
#include <hops/Optimization/ThompsonSampling.hpp>

#include <chrono>
#include <cmath>
#include <memory>
#include <stdexcept>

namespace hops {
    class ExpectedSquaredJumpDistanceTuner {
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
            bool considerTimeCost;
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
                       bool considerTimeCost,
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
        tune(std::vector<std::shared_ptr<MarkovChain>>&, 
             std::vector<RandomNumberGenerator>&, 
             param_type&);

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
             param_type&);

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
             param_type&,
             Eigen::MatrixXd& data,
             Eigen::MatrixXd& posterior);

        ExpectedSquaredJumpDistanceTuner() = delete;
    };

    namespace internal {
        struct ExpectedSquaredJumpDistanceTarget : public ThompsonSamplingTarget<double, Eigen::VectorXd> {
            std::vector<std::shared_ptr<hops::MarkovChain>> markovChain;
            std::vector<RandomNumberGenerator>* randomNumberGenerator;
            ExpectedSquaredJumpDistanceTuner::param_type parameters;

            ExpectedSquaredJumpDistanceTarget(std::vector<std::shared_ptr<hops::MarkovChain>>& markovChain,
                                              std::vector<hops::RandomNumberGenerator>& randomNumberGenerator,
                                              const hops::ExpectedSquaredJumpDistanceTuner::param_type& parameters) :
                    markovChain(markovChain),
                    randomNumberGenerator(&randomNumberGenerator),
                    parameters(parameters) {
                //
            }

            virtual std::tuple<double, double> operator()(const Eigen::VectorXd& x) override;
        };
    }
}

#endif //HOPS_EXPECTEDSQUAREDJUMPDISTANCETUNER_HPP
