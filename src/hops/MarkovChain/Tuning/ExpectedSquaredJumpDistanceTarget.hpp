#ifndef HOPS_EXPECTEDSQUAREDJUMPDISTANCETARGET_HPP
#define HOPS_EXPECTEDSQUAREDJUMPDISTANCETARGET_HPP

#include <hops/MarkovChain/MarkovChain.hpp>
#include <hops/MarkovChain/Tuning/TuningTarget.hpp>
#include <hops/Parallel/OpenMPControls.hpp>
#include <hops/RandomNumberGenerator/RandomNumberGenerator.hpp>
#include <hops/Statistics/ExpectedSquaredJumpDistance.hpp>
#include <hops/Utility/MatrixType.hpp>
#include <hops/Utility/VectorType.hpp>

#include <chrono>
#include <cmath>
#include <memory>
#include <numeric>
#include <stdexcept>

#ifdef _OPENMP
#include <omp.h>
#endif 

namespace hops {
    struct ExpectedSquaredJumpDistanceTarget : public TuningTarget {
        std::vector<std::shared_ptr<MarkovChain>> markovChains;
        unsigned long numberOfTestSamples;
        std::vector<unsigned long> lags;
        bool considerTimeCost;
        bool estimateCovariance;

        ExpectedSquaredJumpDistanceTarget(std::vector<std::shared_ptr<MarkovChain>> markovChains,
                                          unsigned long numberOfTestSamples,
                                          std::vector<unsigned long> lags,
                                          bool considerTimeCost,
                                          bool estimateCovariance) :
            markovChains(markovChains),
            numberOfTestSamples(numberOfTestSamples),
            lags(lags),
            considerTimeCost(considerTimeCost),
            estimateCovariance(estimateCovariance) { }

        std::pair<double, double> operator()(const VectorType& x, const std::vector<RandomNumberGenerator*>& randomNumberGenerators) override;

        std::string getName() const override {
            return "ExpectedSquaredJumpDistance";
        }

        std::unique_ptr<TuningTarget> copyTuningTarget() const override {
            return std::make_unique<ExpectedSquaredJumpDistanceTarget>(*this);
        }
    };

    /**
     * @brief measures the expected squared jump distance of a configured step size
     * @param x
     * @return
     */
    std::pair<double, double> hops::ExpectedSquaredJumpDistanceTarget::operator()(const VectorType& x, const std::vector<RandomNumberGenerator*>& randomNumberGenerators) {
        if (markovChains.size() != randomNumberGenerators.size()) {
            throw std::runtime_error("Number of random number generators must match number of markov chains.");
        }

        double stepSize = std::pow(10, x(0));
        std::vector<double> expectedSquaredJumpDistances(markovChains.size());
        for (size_t i = 0; i < markovChains.size(); ++i) {
            markovChains[i]->setParameter(ProposalParameter::STEP_SIZE, stepSize);
           
            // record time taken to draw samples to scale esjd by time if specified
            unsigned long time = std::chrono::duration_cast<std::chrono::milliseconds>(
                    std::chrono::high_resolution_clock::now().time_since_epoch()
            ).count();
            
            std::vector<VectorType> states(numberOfTestSamples);
            for (size_t j = 0; j < numberOfTestSamples; ++j) {
                states[j] = std::get<1>(markovChains[i]->draw(*randomNumberGenerators[i]));
            }
        
            
            time = std::chrono::duration_cast<std::chrono::milliseconds>(
                    std::chrono::high_resolution_clock::now().time_since_epoch()
            ).count() - time;

            // set time to 1 if it was 0
            time = (time == 0 ? 1 : time);

            // compute covariance upfront to reuse it for higher lag esjds
            MatrixType sqrtCovariance;
            if (estimateCovariance) {
                sqrtCovariance = computeCovariance<VectorType, MatrixType>(states).llt().matrixL();
            } else {
                sqrtCovariance = MatrixType::Identity(states[0].size(), states[0].size());
            }

            double expectedSquaredJumpDistance = 0;

            for (auto& k : lags) {
                expectedSquaredJumpDistance += hops::computeExpectedSquaredJumpDistance<VectorType, MatrixType>(states, sqrtCovariance, k);
            }

            expectedSquaredJumpDistance = (considerTimeCost ? expectedSquaredJumpDistance / time : expectedSquaredJumpDistance);
            expectedSquaredJumpDistances[i] = expectedSquaredJumpDistance;
        }

        double mean = std::accumulate(expectedSquaredJumpDistances.begin(), expectedSquaredJumpDistances.end(), 0.0) / expectedSquaredJumpDistances.size();

        double squaredSum = std::inner_product(expectedSquaredJumpDistances.begin(), expectedSquaredJumpDistances.end(), expectedSquaredJumpDistances.begin(), 0.0);
        double error = squaredSum / expectedSquaredJumpDistances.size() - mean * mean; 

        return {mean, error};
    }
}

#endif // HOPS_EXPECTEDSQUAREDJUMPDISTANCETARGET_HPP
