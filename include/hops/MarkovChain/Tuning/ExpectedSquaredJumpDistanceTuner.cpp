#include <hops/MarkovChain/Tuning/ExpectedSquaredJumpDistanceTuner.hpp>

/**
 * @brief measures the stepsize of a configured step size
 * @param stepSize
 * @param markovChain
 * @return
 */
std::vector<double> hops::internal::ExpectedSquaredJumpDistanceTarget::operator()(const Eigen::VectorXd& x) {
    double stepSize = std::pow(10, x(0));
    std::vector<double> expectedSquaredJumpDistances;
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

        double expectedSquaredJumpDistance = hops::computeExpectedSquaredJumpDistance(markovChain[i]->getStateRecords());
        expectedSquaredJumpDistance = (parameters.considerTimeCost ? expectedSquaredJumpDistance / time : expectedSquaredJumpDistance);
        expectedSquaredJumpDistances.push_back(expectedSquaredJumpDistance);
    }
    return expectedSquaredJumpDistances;
}

bool hops::ExpectedSquaredJumpDistanceTuner::tune(
        double& stepSize,
        double& maximumExpectedSquaredJumpDistance,
        std::vector<std::shared_ptr<hops::MarkovChain>>& markovChain,
        std::vector<RandomNumberGenerator>& randomNumberGenerator,
        const hops::ExpectedSquaredJumpDistanceTuner::param_type& parameters) {
    using Kernel = SquaredExponentialKernel<Eigen::MatrixXd, Eigen::VectorXd>;
    using GP = GaussianProcess<Eigen::MatrixXd, Eigen::VectorXd, Kernel>;

    std::vector<Eigen::VectorXd> logStepSizeGrid;
    double a = std::log10(parameters.stepSizeLowerBound), b = std::log10(parameters.stepSizeUpperBound);

    for (size_t i = 0; i < parameters.stepSizeGridSize; ++i) {
        Eigen::VectorXd x = Eigen::VectorXd(1);
        x(0) = (b - a) * i / (parameters.stepSizeGridSize - 1) + a;
        logStepSizeGrid.push_back(x);
    }

    double sigma = 1, length = 1, noise = 1, unscalingFactor;
    Kernel kernel(sigma, length);
    GP gp = GP(kernel);

    auto target = std::make_shared<internal::ExpectedSquaredJumpDistanceTarget>(internal::ExpectedSquaredJumpDistanceTarget(markovChain, randomNumberGenerator, parameters));

    std::vector<Eigen::VectorXd> samples;
    std::vector<double> observations;

    RandomNumberGenerator thompsonSamplingRandomNumberGenerator(parameters.randomSeed, markovChain.size() + 1);
    bool success = ThompsonSampling<Eigen::MatrixXd, Eigen::VectorXd, GP>::optimize(
            parameters.maximumTotalIterations,
            gp, target, logStepSizeGrid, 
            thompsonSamplingRandomNumberGenerator,
            samples, observations, 
            noise, &unscalingFactor);
   
    auto& posteriorMean = gp.getPosteriorMean();
    auto& posteriorCovariance = gp.getPosteriorCovariance();

    // only for logging purposes
    Eigen::MatrixXd posterior(posteriorMean.size(), 3);
    for (size_t i = 0; i < posteriorMean.size(); ++i) {
        posterior(i, 0) = logStepSizeGrid[i](0);
        posterior(i, 1) =  posteriorMean(i);
        posterior(i, 2) = posteriorCovariance(i,i);
    }

    // only for logging purposes
    Eigen::MatrixXd data(samples.size(), 2);
    for (size_t i = 0; i < samples.size(); ++i) {
        data(i, 0) = samples[i](0);
        data(i, 1) = observations[i];
    }

    // only for logging purposes
	auto tuningDataWriter = FileWriterFactory::createFileWriter(parameters.outputDirectory + "/tuningData", FileWriterType::CSV);
    tuningDataWriter->write("tuner", std::vector<std::string>{"ExpectedSquaredJumpDistanceTuner"});
    tuningDataWriter->write("posterior", posterior);
    tuningDataWriter->write("data", data);

    size_t maximumIndex = 0;
    for (size_t i = 1; i < posteriorMean.size(); ++i) {
        if (posteriorMean(i) > posteriorMean(maximumIndex)) {
            maximumIndex = i;
        }
    }

    stepSize = std::pow(10, logStepSizeGrid[maximumIndex](0));
    maximumExpectedSquaredJumpDistance = posteriorMean(maximumIndex) * unscalingFactor;

    return true;
}

bool hops::ExpectedSquaredJumpDistanceTuner::tune(
        std::vector<std::shared_ptr<hops::MarkovChain>>& markovChain,
        std::vector<RandomNumberGenerator>& randomNumberGenerator,
        const hops::ExpectedSquaredJumpDistanceTuner::param_type& parameters) {
    double stepSize = markovChain[0]->getAttribute(hops::MarkovChainAttribute::STEP_SIZE);
    double maximumExpectedSquaredJumpDistance;
    return tune(stepSize, maximumExpectedSquaredJumpDistance, markovChain, randomNumberGenerator, parameters);
}

hops::ExpectedSquaredJumpDistanceTuner::param_type::param_type(size_t iterationsToTestStepSize,
                                                               size_t maximumTotalIterations,
                                                               size_t stepSizeGridSize,
                                                               double stepSizeLowerBound,
                                                               double stepSizeUpperBound,
                                                               size_t randomSeed,
                                                               bool considerTimeCost,
                                                               std::string outputDirectory) {
    this->iterationsToTestStepSize = iterationsToTestStepSize;
    this->maximumTotalIterations = maximumTotalIterations;
    this->stepSizeGridSize = stepSizeGridSize;
    this->stepSizeLowerBound = stepSizeLowerBound;
    this->stepSizeUpperBound = stepSizeUpperBound;
    this->randomSeed = randomSeed;
    this->considerTimeCost = considerTimeCost;
    this->outputDirectory = outputDirectory;
}

