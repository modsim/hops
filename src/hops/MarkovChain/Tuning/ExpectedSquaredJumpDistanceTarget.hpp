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
#include <stdexcept>

#ifdef _OPENMP
#include <omp.h>
#endif 

namespace hops {
    struct ExpectedSquaredJumpDistanceTarget : public TuningTarget {
        std::vector<std::shared_ptr<MarkovChain>> markovChain;
        std::vector<RandomNumberGenerator*> randomNumberGenerator;
        unsigned long numberOfTestSamples;
        std::vector<unsigned long> lags;
        bool considerTimeCost;

        ExpectedSquaredJumpDistanceTarget() = default;

        ExpectedSquaredJumpDistanceTarget(std::vector<std::shared_ptr<MarkovChain>> markovChain,
                                          std::vector<RandomNumberGenerator*> randomNumberGenerator,
                                          unsigned long numberOfTestSamples,
                                          std::vector<unsigned long> lags,
                                          bool considerTimeCost) :
            markovChain(markovChain),
            randomNumberGenerator(randomNumberGenerator),
            numberOfTestSamples(numberOfTestSamples),
            lags(lags),
            considerTimeCost(considerTimeCost) { }

        std::tuple<double, double> operator()(const VectorType& x) override;

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
    std::tuple<double, double> hops::ExpectedSquaredJumpDistanceTarget::operator()(const VectorType& x) {
        double stepSize = std::pow(10, x(0));
        std::vector<double> expectedSquaredJumpDistances(markovChain.size());
        #pragma omp parallel for num_threads(numberOfThreads)
        for (size_t i = 0; i < markovChain.size(); ++i) {
            markovChain[i]->setParameter(ProposalParameter::STEP_SIZE, stepSize);
           
            // record time taken to draw samples to scale esjd by time if specified
            unsigned long time = std::chrono::duration_cast<std::chrono::milliseconds>(
                    std::chrono::high_resolution_clock::now().time_since_epoch()
            ).count();
            
            std::vector<VectorType> states(numberOfTestSamples);
            for (size_t j = 0; j < numberOfTestSamples; ++j) {
                states[j] = std::get<1>(markovChain[i]->draw(*randomNumberGenerator[i]));
            }
        
            
            time = std::chrono::duration_cast<std::chrono::milliseconds>(
                    std::chrono::high_resolution_clock::now().time_since_epoch()
            ).count() - time;

            // set time to 1 if it was 0
            time = (time == 0 ? 1 : time);

            // compute covariance upfront to reuse it for higher lag esjds
            MatrixType sqrtCovariance = computeCovariance<VectorType, MatrixType>(states).llt().matrixL();

            double expectedSquaredJumpDistance = 0;

            for (auto& k : lags) {
                expectedSquaredJumpDistance += hops::computeExpectedSquaredJumpDistance<VectorType, MatrixType>(states, sqrtCovariance, k);
            }

            expectedSquaredJumpDistance = (considerTimeCost ? expectedSquaredJumpDistance / time : expectedSquaredJumpDistance);
            expectedSquaredJumpDistances[i] = expectedSquaredJumpDistance;
        }

        double mean = std::accumulate(expectedSquaredJumpDistances.begin(), expectedSquaredJumpDistances.end(), 0.0) / expectedSquaredJumpDistances.size();

        double squaredSum = std::inner_product(expectedSquaredJumpDistances.begin(), expectedSquaredJumpDistances.end(), expectedSquaredJumpDistances.begin(), 0.0);
        //double error = std::sqrt(squaredSum / expectedSquaredJumpDistances.size() - mean * mean); 
        double error = squaredSum / expectedSquaredJumpDistances.size() - mean * mean; 

        return {mean, error};
    }
}

#endif // HOPS_EXPECTEDSQUAREDJUMPDISTANCETARGET_HPP
