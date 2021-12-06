#ifndef HOPS_EXPECTEDSQUAREDJUMPDISTANCETARGET_HPP
#define HOPS_EXPECTEDSQUAREDJUMPDISTANCETARGET_HPP

#include <hops/MarkovChain/MarkovChain.hpp>
#include <hops/Parallel/OpenMPControls.hpp>
#include <hops/RandomNumberGenerator/RandomNumberGenerator.hpp>
#include <hops/Statistics/ExpectedSquaredJumpDistance.hpp>

#include <chrono>
#include <cmath>
#include <memory>
#include <stdexcept>

#ifdef _OPENMP
#include <omp.h>
#endif 

namespace hops {
    template<typename StateType, typename MatrixType>
    struct ExpectedSquaredJumpDistanceTarget {
        std::vector<std::shared_ptr<MarkovChain>> markovChain;
        std::vector<RandomNumberGenerator>* randomNumberGenerator;
        unsigned long numberOfTestSamples;
        std::vector<unsigned long> lags;
        bool considerTimeCost;

        std::tuple<double, double> operator()(const StateType& x);

        std::string getName() const {
            return "ExpectedSquaredJumpDistance";
        }
    };

    /**
     * @brief measures the stepsize of a configured step size
     * @param stepSize
     * @param markovChain
     * @return
     */
    template<typename StateType, typename MatrixType>
    std::tuple<double, double> hops::ExpectedSquaredJumpDistanceTarget<StateType, MatrixType>::operator()(const StateType& x) {
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
                states[j] = std::get<1>(markovChain[i]->draw(randomNumberGenerator->at(i)));
            }
        
            
            time = std::chrono::duration_cast<std::chrono::milliseconds>(
                    std::chrono::high_resolution_clock::now().time_since_epoch()
            ).count() - time;

            // set time to 1 if it was 0
            time = (time == 0 ? 1 : time);

            // compute covariance upfront to reuse it for higher lag esjds
            MatrixType sqrtCovariance = computeCovariance<StateType, MatrixType>(states).llt().matrixL();

            double expectedSquaredJumpDistance = 0;

            for (auto& k : lags) {
                expectedSquaredJumpDistance += hops::computeExpectedSquaredJumpDistance<StateType, MatrixType>(states, sqrtCovariance, k);
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
