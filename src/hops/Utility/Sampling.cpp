#include "Sampling.hpp"

#include "hops/MarkovChain/Tuning/AcceptanceRateTarget.hpp"
#include "hops/MarkovChain/Tuning/ThompsonSamplingTuner.hpp"

bool hops::Sampling::tuneChain(hops::RandomNumberGenerator &randomNumberGenerator,
                               const double targetAcceptanceRate,
                               std::shared_ptr<hops::MarkovChain> markovChain,
                               hops::MarkovChainType chainType,
                               hops::FileWriter *fileWriter,
                               double tuningTolerance,
                               long maxTuningIterations) {
    double startStepSize = std::any_cast<double>(markovChain->getParameter(ProposalParameter::STEP_SIZE));
    double lowerLimitStepSize = 1e-2 * startStepSize;
    double upperLimitStepSize = 1e1 * startStepSize;

    size_t iterationsToTestStepSize = 200;
    size_t posteriorUpdateIterations = 100;
    size_t pureSamplingIterations = 10;
    size_t stepSizeGridSize = std::log10(upperLimitStepSize / lowerLimitStepSize) * 10;
    size_t iterationsForConvergence = 5;
    double smoothingLength = 1;
    bool recordData = true;

    std::vector<std::shared_ptr<hops::MarkovChain>> tuningChains = {std::move(markovChain)};
    std::vector<hops::RandomNumberGenerator *> randomNumberGenerators = {&randomNumberGenerator};

    auto tuningTarget = hops::AcceptanceRateTarget(
            tuningChains,
            iterationsToTestStepSize,
            targetAcceptanceRate,
            1);


    hops::MatrixType data;
    double deltaAcceptanceRate = 1;
    hops::VectorType stepSize(1);
    double measuredAcceptanceRate = -1;
    long tuningIteration = 0;
    while (deltaAcceptanceRate > tuningTolerance && tuningIteration < maxTuningIterations) {
        stepSize(0) = std::any_cast<double>(tuningChains[0]->getParameter(hops::ProposalParameter::STEP_SIZE));
        hops::ThompsonSamplingTuner::param_type tuningParameters(
                posteriorUpdateIterations,
                pureSamplingIterations,
                iterationsForConvergence,
                stepSizeGridSize,
                lowerLimitStepSize,
                upperLimitStepSize,
                smoothingLength,
                std::random_device()(),
                recordData);

        hops::ThompsonSamplingTuner::tune(
                stepSize,
                deltaAcceptanceRate,
                randomNumberGenerators,
                tuningParameters,
                tuningTarget,
                data);

        measuredAcceptanceRate = tuningChains[0]->draw(randomNumberGenerator, 2000).first;
        deltaAcceptanceRate = std::abs(targetAcceptanceRate - measuredAcceptanceRate);

        std::stringstream stream;
        stream << "tuning iter: " << tuningIteration << " " << hops::markovChainTypeToShortString(chainType) << " s: "
               << stepSize(0) << " alpha: "
               << measuredAcceptanceRate
               << " (delta: " << deltaAcceptanceRate << ")" << " u: " << upperLimitStepSize << " l: "
               << lowerLimitStepSize << std::endl;

        fileWriter->write("tuning_debug_info", std::vector<std::string>{stream.str()});


        // Does not tune step size too high for our hit&run walks.
        if (chainType == hops::MarkovChainType::CoordinateHitAndRun || chainType == hops::MarkovChainType::HitAndRun) {
            if (upperLimitStepSize >= 10) {
                break;
            }
        }

        if (deltaAcceptanceRate > tuningTolerance && measuredAcceptanceRate < targetAcceptanceRate) {
            upperLimitStepSize /= 2;
            lowerLimitStepSize /= 2;
        }
        if (deltaAcceptanceRate > tuningTolerance && measuredAcceptanceRate > targetAcceptanceRate) {
            upperLimitStepSize *= 2;
            lowerLimitStepSize *= 2;
        }
        stepSizeGridSize += std::log10(upperLimitStepSize / lowerLimitStepSize);
        tuningIteration++;
    }

    return deltaAcceptanceRate <= tuningTolerance;
}
