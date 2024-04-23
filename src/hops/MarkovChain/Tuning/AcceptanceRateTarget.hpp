#ifndef HOPS_ACCEPTANCERATETARGET_HPP
#define HOPS_ACCEPTANCERATETARGET_HPP

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
#include <utility>

#ifdef _OPENMP
#include <omp.h>
#endif 

namespace hops {
    struct AcceptanceRateTarget : public TuningTarget {
        std::vector<std::shared_ptr<MarkovChain>> markovChains;
        unsigned long numberOfTestSamples;
        double acceptanceRateTargetValue;
        unsigned long order;

        AcceptanceRateTarget(std::vector<std::shared_ptr<MarkovChain>> markovChains,
                                          unsigned long numberOfTestSamples,
                                          double acceptanceRateTargetValue,
                                          unsigned long order = 1);


        [[nodiscard]] std::string getName() const override;

        [[nodiscard]] std::unique_ptr<TuningTarget> copyTuningTarget() const override;

        std::pair<double, double>
        operator()(const VectorType &x, const std::vector<RandomNumberGenerator *> &randomNumberGenerators) override;
    };

} // namespace hops

#endif // HOPS_ACCEPTANCERATETARGET_HPP

