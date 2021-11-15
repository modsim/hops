#include "SimpleExpectedSquaredJumpDistanceTuner.hpp"

//extern std::vector<double> measureExpectedSquaredJumpDistance(double stepSize,
//                                                              std::vector<std::shared_ptr<hops::MarkovChain>>& markovChain,
//                                                              std::vector<hops::RandomNumberGenerator>& randomNumberGenerator,
//                                                              const hops::SimpleExpectedSquaredJumpDistanceTuner::param_type& parameters);

std::vector<double> measureExpectedSquaredJumpDistance(double stepSize,
                                           std::vector<std::shared_ptr<hops::MarkovChain>>& markovChain,
                                           std::vector<hops::RandomNumberGenerator>& randomNumberGenerator,
                                           const hops::SimpleExpectedSquaredJumpDistanceTuner::param_type& parameters) {
    std::vector<double> expectedSquaredJumpDistances;

    for (size_t i = 0; i < markovChain.size(); ++i) {
        markovChain[i]->clearHistory();
        markovChain[i]->setAttribute(hops::MarkovChainAttribute::STEP_SIZE, stepSize);
       
        // record time taken to draw samples to scale esjd by time if specified
        unsigned long time = std::chrono::duration_cast<std::chrono::milliseconds>(
                std::chrono::high_resolution_clock::now().time_since_epoch()
        ).count();
        
        markovChain[i]->draw(randomNumberGenerator[i], parameters.iterationsToTestStepSize);
        
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

    return expectedSquaredJumpDistances;
}

bool hops::SimpleExpectedSquaredJumpDistanceTuner::tune(
        double& stepSize,
        double& maximumExpectedSquaredJumpDistance,
        std::vector<std::shared_ptr<hops::MarkovChain>>& markovChain,
        std::vector<RandomNumberGenerator>& randomNumberGenerator,
        const hops::SimpleExpectedSquaredJumpDistanceTuner::param_type& parameters) {
    std::vector<Eigen::VectorXd> logStepSizeGrid;
    std::vector<double> meanObservedExpectedSquaredJumpDistances(parameters.stepSizeGridSize, 0);
    double a = std::log10(parameters.stepSizeLowerBound), b = std::log10(parameters.stepSizeUpperBound);

    for (size_t i = 0; i < parameters.stepSizeGridSize; ++i) {
        Eigen::VectorXd x = Eigen::VectorXd(1);
        x(0) = (b - a) * i / (parameters.stepSizeGridSize - 1) + a;
        logStepSizeGrid.push_back(x);
    }

    double maximumObservedExpectedSquaredJumpDistance = 0;
    size_t maxElementIndex;

    // only for logging purposes
    Eigen::MatrixXd data(parameters.stepSizeGridSize, 2);

	for (size_t i = 0; i < parameters.stepSizeGridSize; ++i) {
        auto testStepSize = logStepSizeGrid[i];
        auto evaluations = measureExpectedSquaredJumpDistance(std::pow(10, testStepSize(0)), markovChain, randomNumberGenerator, parameters);

        double mean = 0;
        for (size_t j = 0; j < evaluations.size(); ++j) {
            mean += evaluations[j];
        }
        mean /= evaluations.size();

        // only for logging purposes
        data(i, 0) = testStepSize(0);
        data(i, 1) = mean;

        if (mean > maximumObservedExpectedSquaredJumpDistance) {
            maximumObservedExpectedSquaredJumpDistance = mean;
            maxElementIndex = i;
        }
    }

    // only for logging purposes
	auto tuningDataWriter = hops::FileWriterFactory::createFileWriter(parameters.outputDirectory + "/tuningData", FileWriterType::CSV);
    tuningDataWriter->write("tuner", std::vector<std::string>{"SimpleExpectedSquaredJumpDistanceTuner"});
    tuningDataWriter->write("data", data);

    maximumExpectedSquaredJumpDistance = maximumObservedExpectedSquaredJumpDistance;
    stepSize = std::pow(10, (logStepSizeGrid[maxElementIndex](0)));
    
    return true;
}

bool hops::SimpleExpectedSquaredJumpDistanceTuner::tune(
        std::vector<std::shared_ptr<hops::MarkovChain>>& markovChain,
        std::vector<RandomNumberGenerator>& randomNumberGenerator,
        const hops::SimpleExpectedSquaredJumpDistanceTuner::param_type& parameters) {
    double stepSize = markovChain[0]->getAttribute(hops::MarkovChainAttribute::STEP_SIZE);
    double maximumExpectedSquaredJumpDistance;
    return tune(stepSize, maximumExpectedSquaredJumpDistance, markovChain, randomNumberGenerator, parameters);
}

hops::SimpleExpectedSquaredJumpDistanceTuner::param_type::param_type(size_t iterationsToTestStepSize,
                                                               //size_t maximumTotalIterations,
                                                               size_t stepSizeGridSize,
                                                               double stepSizeLowerBound,
                                                               double stepSizeUpperBound,
                                                               bool considerTimeCost,
                                                               std::string outputDirectory) {
    this->iterationsToTestStepSize = iterationsToTestStepSize;
    //this->maximumTotalIterations = maximumTotalIterations;
    this->stepSizeGridSize = stepSizeGridSize;
    this->stepSizeLowerBound = stepSizeLowerBound;
    this->stepSizeUpperBound = stepSizeUpperBound;
    this->considerTimeCost = considerTimeCost;
    this->outputDirectory = outputDirectory;
}

