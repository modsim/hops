#include "ExpectedSquaredJumpDistanceTuner.hpp"

bool hops::ExpectedSquaredJumpDistanceTuner::tune(
        VectorType& stepSize,
        double& maximumExpectedSquaredJumpDistance,
        std::vector<std::shared_ptr<hops::MarkovChain>>& markovChains,
        const std::vector<RandomNumberGenerator*>& randomNumberGenerators,
        ExpectedSquaredJumpDistanceTuner::param_type& parameters,
        Eigen::MatrixXd& data) {
    auto target = ExpectedSquaredJumpDistanceTarget{markovChains, 
                                                    static_cast<unsigned long>(parameters.iterationsToTestStepSize), 
                                                    /*lags=*/{1},
                                                    parameters.considerTimeCost,
                                                    parameters.estimateCovariance};

    return ThompsonSamplingTuner::tune(stepSize, maximumExpectedSquaredJumpDistance, randomNumberGenerators, parameters.ts_params, target, data);
}

bool hops::ExpectedSquaredJumpDistanceTuner::tune(
        std::vector<std::shared_ptr<hops::MarkovChain>>& markovChains,
        const std::vector<RandomNumberGenerator*>& randomNumberGenerators,
        hops::ExpectedSquaredJumpDistanceTuner::param_type& parameters) {
    VectorType stepSize = std::any_cast<double>(markovChains[0]->getParameter(ProposalParameter::STEP_SIZE)) * VectorType::Ones(1);
    double maximumExpectedSquaredJumpDistance;
    return tune(stepSize, maximumExpectedSquaredJumpDistance, markovChains, randomNumberGenerators, parameters);
}

bool hops::ExpectedSquaredJumpDistanceTuner::tune(
        VectorType& stepSize,
        double& maximumExpectedSquaredJumpDistance,
        std::vector<std::shared_ptr<hops::MarkovChain>>& markovChains,
        const std::vector<RandomNumberGenerator*>& randomNumberGenerators,
        hops::ExpectedSquaredJumpDistanceTuner::param_type& parameters) {
    Eigen::MatrixXd data;
    return tune(stepSize, maximumExpectedSquaredJumpDistance, markovChains, randomNumberGenerators, parameters, data);
}

hops::ExpectedSquaredJumpDistanceTuner::param_type::param_type(size_t iterationsToTestStepSize,
                                                               size_t posteriorUpdateIterations,
                                                               size_t pureSamplingIterations,
                                                               size_t iterationsForConvergence,
                                                               size_t stepSizeGridSize,
                                                               double stepSizeLowerBound,
                                                               double stepSizeUpperBound,
                                                               double smoothingLength,
                                                               size_t randomSeed,
                                                               bool recordData,
                                                               std::vector<unsigned long> lags,
                                                               bool considerTimeCost,
                                                               bool estimateCovariance) {
    this->iterationsToTestStepSize = iterationsToTestStepSize;
    this->ts_params.posteriorUpdateIterations = posteriorUpdateIterations;
    this->ts_params.pureSamplingIterations = pureSamplingIterations;
    this->ts_params.iterationsForConvergence = iterationsForConvergence;
    this->ts_params.posteriorUpdateIterationsNeeded = 0;
    this->ts_params.stepSizeGridSize = stepSizeGridSize;
    this->ts_params.stepSizeLowerBound = stepSizeLowerBound;
    this->ts_params.stepSizeUpperBound = stepSizeUpperBound;
    this->ts_params.smoothingLength = smoothingLength;
    this->ts_params.randomSeed = randomSeed;
    this->ts_params.recordData = recordData;
    this->lags = lags;
    this->considerTimeCost = considerTimeCost;
    this->estimateCovariance = estimateCovariance;
}

