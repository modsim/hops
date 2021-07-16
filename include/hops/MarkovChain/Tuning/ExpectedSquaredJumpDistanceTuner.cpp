#include <hops/MarkovChain/Tuning/ExpectedSquaredJumpDistanceTuner.hpp>

/**
 * @brief measures the stepsize of a configured step size
 * @param stepSize
 * @param markovChain
 * @return
 */
std::tuple<double, double> hops::internal::ExpectedSquaredJumpDistanceTarget::operator()(const Eigen::VectorXd& x) {
    double stepSize = std::pow(10, x(0));
    std::vector<double> expectedSquaredJumpDistances;
    #pragma omp parallel for
    for (size_t i = 0; i < markovChain.size(); ++i) {
        markovChain[i]->clearHistory();
        markovChain[i]->setAttribute(hops::MarkovChainAttribute::STEP_SIZE, stepSize);
       
        // record time taken to draw samples to scale esjd by time if specified
        unsigned long time = std::chrono::duration_cast<std::chrono::milliseconds>(
                std::chrono::high_resolution_clock::now().time_since_epoch()
        ).count();
        
        markovChain[i]->draw(randomNumberGenerator->at(i), parameters.iterationsToTestStepSize);
        
        time = std::chrono::duration_cast<std::chrono::milliseconds>(
                std::chrono::high_resolution_clock::now().time_since_epoch()
        ).count() - time;

        // set time to 1 if it was 0
        time = (time == 0 ? 1 : time);

        double expectedSquaredJumpDistance = 
                hops::computeExpectedSquaredJumpDistance<Eigen::VectorXd, Eigen::MatrixXd>(markovChain[i]->getStateRecords());

        expectedSquaredJumpDistance = (parameters.considerTimeCost ? expectedSquaredJumpDistance / time : expectedSquaredJumpDistance);
        expectedSquaredJumpDistances.push_back(expectedSquaredJumpDistance);
    }

    double mean = std::accumulate(expectedSquaredJumpDistances.begin(), expectedSquaredJumpDistances.end(), 0.0) / expectedSquaredJumpDistances.size();

    double squaredSum = std::inner_product(expectedSquaredJumpDistances.begin(), expectedSquaredJumpDistances.end(), expectedSquaredJumpDistances.begin(), 0.0);
    double error = std::sqrt(squaredSum / expectedSquaredJumpDistances.size() - mean * mean); 

    return {mean, error};
}

bool hops::ExpectedSquaredJumpDistanceTuner::tune(
        double& stepSize,
        double& maximumExpectedSquaredJumpDistance,
        std::vector<std::shared_ptr<hops::MarkovChain>>& markovChain,
        std::vector<RandomNumberGenerator>& randomNumberGenerator,
        hops::ExpectedSquaredJumpDistanceTuner::param_type& parameters,
        Eigen::MatrixXd& data,
        Eigen::MatrixXd& posterior) {
    using Kernel = SquaredExponentialKernel<Eigen::MatrixXd, Eigen::VectorXd>;
    using GP = GaussianProcess<Eigen::MatrixXd, Eigen::VectorXd, Kernel>;

    Eigen::VectorXd logStepSizeGrid(parameters.stepSizeGridSize);
    double a = std::log10(parameters.stepSizeLowerBound), b = std::log10(parameters.stepSizeUpperBound);

    for (size_t i = 0; i < parameters.stepSizeGridSize; ++i) {
        logStepSizeGrid(i) = (b - a) * i / (parameters.stepSizeGridSize - 1) + a;
    }

    double sigma = 1, length = 1;
    Kernel kernel(sigma, length);
    GP gp = GP(kernel);

    auto target = std::make_shared<internal::ExpectedSquaredJumpDistanceTarget>(
            internal::ExpectedSquaredJumpDistanceTarget(markovChain, randomNumberGenerator, parameters));

    RandomNumberGenerator thompsonSamplingRandomNumberGenerator(parameters.randomSeed, markovChain.size() + 1);
    bool isThompsonSamplingConverged = ThompsonSampling<Eigen::MatrixXd, Eigen::VectorXd, GP>::optimize(
            parameters.posteriorUpdateIterations,
            parameters.pureSamplingIterations,
            parameters.iterationsForConvergence,
            gp, target, logStepSizeGrid, 
            thompsonSamplingRandomNumberGenerator,
            &parameters.posteriorUpdateIterationsNeeded,
            parameters.smoothingLength);
   
    if (parameters.recordData) {
        auto& posteriorMean = gp.getPosteriorMean();
        auto& posteriorCovariance = gp.getPosteriorCovariance();

        auto& observedInputs = gp.getObservedInputs();
        auto& observedValues = gp.getObservedValues();
        auto& observedValueErrors = gp.getObservedValueErrors();

        // only for logging purposes
        posterior = Eigen::MatrixXd(posteriorMean.size(), 3);
        for (long i = 0; i < posteriorMean.size(); ++i) {
            posterior(i, 0) = logStepSizeGrid(i, 0);
            posterior(i, 1) = posteriorMean(i);
            posterior(i, 2) = posteriorCovariance(i,i);
        }

        // only for logging purposes
        data = Eigen::MatrixXd(observedInputs.size(), 3);
        for (long i = 0; i < observedInputs.size(); ++i) {
            data(i, 0) = observedInputs(i, 0);
            data(i, 1) = observedValues(i);
            data(i, 2) = observedValueErrors(i);
        }
    }

    // store results in reference parameters
    auto& posteriorMean = gp.getPosteriorMean();
    size_t maximumIndex;
    maximumExpectedSquaredJumpDistance = posteriorMean.maxCoeff(&maximumIndex);
    stepSize = std::pow(10, logStepSizeGrid(maximumIndex, 0));

    return isThompsonSamplingConverged;
}

bool hops::ExpectedSquaredJumpDistanceTuner::tune(
        std::vector<std::shared_ptr<hops::MarkovChain>>& markovChain,
        std::vector<RandomNumberGenerator>& randomNumberGenerator,
        hops::ExpectedSquaredJumpDistanceTuner::param_type& parameters) {
    double stepSize = markovChain[0]->getAttribute(hops::MarkovChainAttribute::STEP_SIZE);
    double maximumExpectedSquaredJumpDistance;
    return tune(stepSize, maximumExpectedSquaredJumpDistance, markovChain, randomNumberGenerator, parameters);
}

bool hops::ExpectedSquaredJumpDistanceTuner::tune(
        double& stepSize,
        double& maximumExpectedSquaredJumpDistance,
        std::vector<std::shared_ptr<hops::MarkovChain>>& markovChain,
        std::vector<RandomNumberGenerator>& randomNumberGenerator,
        hops::ExpectedSquaredJumpDistanceTuner::param_type& parameters) {
    Eigen::MatrixXd data, posterior;
    return tune(stepSize, maximumExpectedSquaredJumpDistance, markovChain, randomNumberGenerator, parameters, data, posterior);
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
    this->posteriorUpdateIterations = posteriorUpdateIterations;
    this->pureSamplingIterations = pureSamplingIterations;
    this->iterationsForConvergence = iterationsForConvergence;
    this->posteriorUpdateIterationsNeeded = 0;
    this->stepSizeGridSize = stepSizeGridSize;
    this->stepSizeLowerBound = stepSizeLowerBound;
    this->stepSizeUpperBound = stepSizeUpperBound;
    this->smoothingLength = smoothingLength;
    this->randomSeed = randomSeed;
    this->considerTimeCost = considerTimeCost;
    this->recordData = recordData;
}

