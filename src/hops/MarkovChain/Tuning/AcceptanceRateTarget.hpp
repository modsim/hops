#ifndef HOPS_ACCEPTANCERATETARGET_HPP
#define HOPS_ACCEPTANCERATETARGET_HPP

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
    struct AcceptanceRateTarget : public TuningTarget {
        std::vector<std::shared_ptr<MarkovChain>> markovChains;
        unsigned long numberOfTestSamples;
        double acceptanceRateTargetValue;

        AcceptanceRateTarget(std::vector<std::shared_ptr<MarkovChain>> markovChains,
                                          unsigned long numberOfTestSamples,
                                          double acceptanceRateTargetValue) :
            markovChains(markovChains),
            numberOfTestSamples(numberOfTestSamples),
            acceptanceRateTargetValue(acceptanceRateTargetValue) { }

        std::pair<double, double> operator()(const VectorType& x, const std::vector<RandomNumberGenerator*>& randomNumberGenerators) override;

        std::string getName() const override {
            return "AcceptanceRate";
        }

        std::unique_ptr<TuningTarget> copyTuningTarget() const override {
            return std::make_unique<AcceptanceRateTarget>(*this);
        }
    };

    std::pair<double, double> hops::AcceptanceRateTarget::operator()(const VectorType& x, const std::vector<RandomNumberGenerator*>& randomNumberGenerators) {
        if (markovChains.size() != randomNumberGenerators.size()) {
            throw std::runtime_error("Number of random number generators must match number of markov chains.");
        }

        double stepSize = std::pow(10, x(0));
        std::vector<double> acceptanceRateScores(markovChains.size());
        #pragma omp parallel for num_threads(numberOfThreads)
        for (size_t i = 0; i < markovChains.size(); ++i) {
            markovChains[i]->setParameter(ProposalParameter::STEP_SIZE, stepSize);

            double acceptanceRate = std::get<0>(markovChains[i]->draw(*randomNumberGenerators[i], numberOfTestSamples));

            double deltaScale = (
                    acceptanceRate > acceptanceRateTargetValue ?
                    1 - acceptanceRateTargetValue :
                    acceptanceRateTargetValue
            );
            acceptanceRateScores[i] = 1 - std::abs(acceptanceRate - acceptanceRateTargetValue) / deltaScale;
        }

        double mean = std::accumulate(acceptanceRateScores.begin(), acceptanceRateScores.end(), 0.0) / acceptanceRateScores.size();

        double squaredSum = std::inner_product(acceptanceRateScores.begin(), acceptanceRateScores.end(), acceptanceRateScores.begin(), 0.0);
        double error = squaredSum / acceptanceRateScores.size() - mean * mean; 

        return {mean, error};
    }
} // namespace hops

#endif // HOPS_ACCEPTANCERATETARGET_HPP

