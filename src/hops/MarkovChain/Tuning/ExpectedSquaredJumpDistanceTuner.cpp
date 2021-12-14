#include "ExpectedSquaredJumpDistanceTuner.hpp"

bool hops::ExpectedSquaredJumpDistanceTuner::tune(
        VectorType& stepSize,
        double& maximumExpectedSquaredJumpDistance,
        std::vector<std::shared_ptr<hops::MarkovChain>>& markovChains,
        const std::vector<RandomNumberGenerator*>& randomNumberGenerators,
        ExpectedSquaredJumpDistanceTuner::param_type& parameters,
        Eigen::MatrixXd& data) {
    auto target = ExpectedSquaredJumpDistanceTarget{markovChains, 
                                                    parameters.iterationsToTestStepSize, 
                                                    /*lags=*/{1},
                                                    parameters.considerTimeCost};

    return ThompsonSamplingTuner::tune(stepSize, maximumExpectedSquaredJumpDistance, randomNumberGenerators, parameters.ts_params, target, data);
    //using Kernel = SquaredExponentialKernel<Eigen::MatrixXd, Eigen::VectorXd>;
    //using GP = GaussianProcess<Eigen::MatrixXd, Eigen::VectorXd, Kernel>;

    //Eigen::VectorXd logStepSizeGrid(parameters.stepSizeGridSize);
    //double a = std::log10(parameters.stepSizeLowerBound), b = std::log10(parameters.stepSizeUpperBound);

    //for (size_t i = 0; i < parameters.stepSizeGridSize; ++i) {
    //    logStepSizeGrid(i) = (b - a) * i / (parameters.stepSizeGridSize - 1) + a;
    //}

    //double sigma = 1, length = 1;
    //Kernel kernel(sigma, length);
    //GP gp = GP(kernel);

    //auto target = ExpectedSquaredJumpDistanceTarget{markovChains, 
    //                                                parameters.iterationsToTestStepSize, 
    //                                                /*lags=*/{1},
    //                                                parameters.considerTimeCost};

    //RandomNumberGenerator thompsonSamplingRandomNumberGenerator(parameters.randomSeed, randomNumberGenerators.size() + 1);
    //bool isThompsonSamplingConverged = ThompsonSampling<GP, decltype(target)>::optimize(
    //        parameters.posteriorUpdateIterations,
    //        parameters.pureSamplingIterations,
    //        parameters.iterationsForConvergence,
    //        gp, target, logStepSizeGrid, 
    //        randomNumberGenerators,
    //        thompsonSamplingRandomNumberGenerator,
    //        &parameters.posteriorUpdateIterationsNeeded,
    //        parameters.smoothingLength);
   
    //if (parameters.recordData) {
    //    auto& posteriorMean = gp.getPosteriorMean();
    //    auto& posteriorCovariance = gp.getPosteriorCovariance();

    //    auto& observedInputs = gp.getObservedInputs();
    //    auto& observedValues = gp.getObservedValues();
    //    auto& observedValueErrors = gp.getObservedValueErrors();

    //    // only for logging purposes
    //    posterior = Eigen::MatrixXd(posteriorMean.size(), 3);
    //    for (long i = 0; i < posteriorMean.size(); ++i) {
    //        posterior(i, 0) = logStepSizeGrid(i, 0);
    //        posterior(i, 1) = posteriorMean(i);
    //        posterior(i, 2) = posteriorCovariance(i,i);
    //    }

    //    // only for logging purposes
    //    data = Eigen::MatrixXd(observedInputs.size(), 3);
    //    for (long i = 0; i < observedInputs.size(); ++i) {
    //        data(i, 0) = observedInputs(i, 0);
    //        data(i, 1) = observedValues(i);
    //        data(i, 2) = observedValueErrors(i);
    //    }
    //}

    //// store results in reference parameters
    //auto& posteriorMean = gp.getPosteriorMean();
    //size_t maximumIndex;
    //maximumExpectedSquaredJumpDistance = posteriorMean.maxCoeff(&maximumIndex);
    //stepSize = std::pow(10, logStepSizeGrid(maximumIndex, 0));

    //return isThompsonSamplingConverged;
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
                                                               bool considerTimeCost,
                                                               bool recordData) {
    this->iterationsToTestStepSize = iterationsToTestStepSize;
    //this->ts_params.iterationsToTestStepSize = iterationsToTestStepSize;
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
    this->considerTimeCost = considerTimeCost;
}

