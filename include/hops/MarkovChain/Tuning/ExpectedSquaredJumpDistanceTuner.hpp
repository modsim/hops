#ifndef NEW_HOPS_EXPECTEDSQUAREDJUMPDISTANCETUNER_HPP
#define NEW_HOPS_EXPECTEDSQUAREDJUMPDISTANCETUNER_HPP

#include <hops/MarkovChain/MarkovChain.hpp>
#include <hops/Optimization/ThompsonSampling.hpp>
#include <hops/Diagnostics/ExpectedSquaredJumpDistance.hpp>
#include <memory>
#include <chrono>
#include <stdexcept>

namespace hops {
    class ExpectedSquaredJumpDistanceTuner {
    public:
        struct param_type {
            size_t iterationsToTestStepSize;
            size_t maximumTotalIterations;
            size_t stepSizeGridSize;
            double stepSizeLowerBound;
            double stepSizeUpperBound;
            size_t randomSeed;
            bool considerTimeCost;

            param_type(size_t iterationsToTestStepSize,
                       size_t maximumTotalIterations,
                       size_t stepSizeGridSize,
                       double stepSizeLowerBound,
                       double stepSizeUpperBound,
                       size_t randomSeed,
                       bool considerTimeCost
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

        ExpectedSquaredJumpDistanceTuner() = delete;
    };

    namespace internal {
        struct ExpectedSquaredJumpDistanceTarget : public ThompsonSamplingTarget<std::vector<double>, Eigen::VectorXd> {
            std::vector<std::shared_ptr<hops::MarkovChain>> markovChain;
            std::shared_ptr<std::vector<RandomNumberGenerator>> randomNumberGenerator;
            ExpectedSquaredJumpDistanceTuner::param_type parameters;

            ExpectedSquaredJumpDistanceTarget(std::vector<std::shared_ptr<hops::MarkovChain>>& markovChain,
                                              std::vector<hops::RandomNumberGenerator>& randomNumberGenerator,
                                              const hops::ExpectedSquaredJumpDistanceTuner::param_type& parameters) :
                    markovChain(markovChain),
                    randomNumberGenerator(&randomNumberGenerator),
                    parameters(parameters) {
                //
            }

            virtual std::vector<double> operator()(const Eigen::VectorXd& x) override;
        };
    }
}

#endif //HOPS_EXPECTEDSQUAREDJUMPDISTANCETUNER_HPP
