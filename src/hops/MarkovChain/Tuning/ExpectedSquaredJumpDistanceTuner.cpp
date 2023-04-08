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

