#ifndef HOPS_EXPECTEDSQUAREDJUMPDISTANCETARGET_HPP
#define HOPS_EXPECTEDSQUAREDJUMPDISTANCETARGET_HPP

#include "hops/MarkovChain/MarkovChain.hpp"
#include "hops/MarkovChain/Tuning/TuningTarget.hpp"
#include "hops/RandomNumberGenerator/RandomNumberGenerator.hpp"
#include "hops/Statistics/ExpectedSquaredJumpDistance.hpp"
#include "hops/Utility/MatrixType.hpp"
#include "hops/Utility/VectorType.hpp"

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

        /**
         * @brief measures the expected squared jump distance of a configured step size
         * @param x
         * @return
         */
        std::pair<double, double> operator()(const VectorType& x, const std::vector<RandomNumberGenerator*>& randomNumberGenerators) override;

        std::string getName() const override {
            return "ExpectedSquaredJumpDistance";
        }

        std::unique_ptr<TuningTarget> copyTuningTarget() const override {
            return std::make_unique<ExpectedSquaredJumpDistanceTarget>(*this);
        }
    };

}

#endif // HOPS_EXPECTEDSQUAREDJUMPDISTANCETARGET_HPP
