#ifndef HOPS_ACCEPTANCERATETARGET_HPP
#define HOPS_ACCEPTANCERATETARGET_HPP

#include <hops/MarkovChain/MarkovChain.hpp>
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
    template<typename StateType>
    struct AcceptanceRateTarget {
        std::vector<std::shared_ptr<MarkovChain>> markovChain;
        std::vector<RandomNumberGenerator>* randomNumberGenerator;
        unsigned long numberOfTestSamples;
        double acceptanceRateTargetValue;

        std::tuple<double, double> operator()(const StateType& x);

        std::string getName() const {
            return "AcceptanceRate";
        }
    };

    template<typename StateType>
    std::tuple<double, double> hops::AcceptanceRateTarget<StateType>::operator()(const StateType& x) {
        double stepSize = std::pow(10, x(0));
        std::vector<double> acceptanceRateScores(markovChain.size());
        #pragma omp parallel for
        for (size_t i = 0; i < markovChain.size(); ++i) {
            markovChain[i]->clearHistory();
            markovChain[i]->setAttribute(hops::MarkovChainAttribute::STEP_SIZE, stepSize);

            markovChain[i]->draw(randomNumberGenerator->at(i), numberOfTestSamples);

            double acceptanceRate = markovChain[i]->getAcceptanceRate();
            double deltaScale = (
                    acceptanceRate > acceptanceRateTargetValue ?
                    1 - acceptanceRateTargetValue :
                    acceptanceRateTargetValue
            );
            acceptanceRateScores[i] = 1 - std::abs(acceptanceRate - acceptanceRateTargetValue) / deltaScale;
        }

        double mean = std::accumulate(acceptanceRateScores.begin(), acceptanceRateScores.end(), 0.0) / acceptanceRateScores.size();

        double squaredSum = std::inner_product(acceptanceRateScores.begin(), acceptanceRateScores.end(), acceptanceRateScores.begin(), 0.0);
        //double error = std::sqrt(squaredSum / acceptanceRateScores.size() - mean * mean); 
        double error = squaredSum / acceptanceRateScores.size() - mean * mean; 

        return {mean, error};
    }
} // namespace hops

#endif // HOPS_ACCEPTANCERATETARGET_HPP

