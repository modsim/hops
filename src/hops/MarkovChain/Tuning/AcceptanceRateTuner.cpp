#include "AcceptanceRateTuner.hpp"

bool hops::AcceptanceRateTuner::tune(
        VectorType& stepSize,
        double& deltaAcceptanceRate,
        std::vector<std::shared_ptr<hops::MarkovChain>>& markovChains,
        const std::vector<RandomNumberGenerator*>& randomNumberGenerators,
        AcceptanceRateTuner::param_type &parameters,
        Eigen::MatrixXd& data) {
    auto target = AcceptanceRateTarget(markovChains, parameters.iterationsToTestStepSize, parameters.acceptanceRateTargetValue);
    return ThompsonSamplingTuner::tune(stepSize, deltaAcceptanceRate, randomNumberGenerators, parameters.ts_params, target, data);
}

bool hops::AcceptanceRateTuner::tune(
        std::vector<std::shared_ptr<hops::MarkovChain>> &markovChains,
        const std::vector<RandomNumberGenerator*> &randomNumberGenerators,
        hops::AcceptanceRateTuner::param_type &parameters) {
    VectorType stepSize = std::any_cast<double>(markovChains[0]->getParameter(ProposalParameter::STEP_SIZE)) * VectorType::Ones(1);
    double deltaAcceptanceRate;
    return tune(stepSize, deltaAcceptanceRate, markovChains, randomNumberGenerators, parameters);
}

bool hops::AcceptanceRateTuner::tune(
        VectorType &stepSize,
        double &deltaAcceptanceRate,
        std::vector<std::shared_ptr<hops::MarkovChain>> &markovChains,
        const std::vector<RandomNumberGenerator*> &randomNumberGenerators,
        hops::AcceptanceRateTuner::param_type &parameters) {
    Eigen::MatrixXd data;
    return tune(stepSize, deltaAcceptanceRate, markovChains, randomNumberGenerators, parameters, data);
}

hops::AcceptanceRateTuner::param_type::param_type(double acceptanceRateTargetValue,
                                                  size_t iterationsToTestStepSize,
                                                  size_t posteriorUpdateIterations,
                                                  size_t pureSamplingIterations,
                                                  size_t iterationsForConvergence,
                                                  size_t stepSizeGridSize,
                                                  double stepSizeLowerBound,
                                                  double stepSizeUpperBound,
                                                  double smoothingLength,
                                                  size_t randomSeed,
                                                  bool recordData) {
    this->acceptanceRateTargetValue = acceptanceRateTargetValue;
    this->iterationsToTestStepSize = iterationsToTestStepSize;
    this->ts_params.posteriorUpdateIterations = posteriorUpdateIterations;
    this->ts_params.pureSamplingIterations = pureSamplingIterations;
    this->ts_params.iterationsForConvergence = iterationsForConvergence;
    this->ts_params.posteriorUpdateIterationsNeeded = 0;
    this->ts_params.stepSizeGridSize = stepSizeGridSize;
    this->ts_params.stepSizeLowerBound = stepSizeLowerBound;
    this->ts_params.stepSizeUpperBound = stepSizeUpperBound;
    this->ts_params.smoothingLength = smoothingLength;
    this->ts_params.recordData = recordData;
    this->ts_params.randomSeed = randomSeed;
}

