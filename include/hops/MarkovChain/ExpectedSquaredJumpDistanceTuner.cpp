#include <hops/MarkovChain/MarkovChain.hpp>
#include <hops/MarkovChain/MarkovChainAttribute.hpp>
#include <hops/MarkovChain/ExpectedSquaredJumpDistanceTuner.hpp>
#include <hops/Optimization/GaussianProcess.hpp>

#include <cmath>
#include <memory>
#include <chrono>

/**
 * @brief measures the stepsize of a configured step size
 * @param stepSize
 * @param markovChain
 * @return
 */
std::vector<double> measureExpectedSquaredJumpDistance(double stepSize,
                                           std::vector<std::shared_ptr<hops::MarkovChain>>& markovChain,
                                           std::vector<hops::RandomNumberGenerator>& randomNumberGenerator,
                                           const hops::ExpectedSquaredJumpDistanceTuner::param_type& parameters) {
    std::vector<double> expectedSquaredJumpDistances;
    for (size_t i = 0; i < markovChain.size(); ++i) {
        markovChain[i]->clearHistory();
        markovChain[i]->setAttribute(hops::MarkovChainAttribute::STEP_SIZE, stepSize);
       
        // record time taken to draw samples to scale esjd by time if specified
        double time = std::chrono::duration_cast<std::chrono::milliseconds>(
                std::chrono::high_resolution_clock::now().time_since_epoch()
        ).count();
        
        markovChain[i]->draw(randomNumberGenerator[i], parameters.iterationsToTestStepSize);
        
        time = std::chrono::duration_cast<std::chrono::milliseconds>(
                std::chrono::high_resolution_clock::now().time_since_epoch()
        ).count() - time;

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

    std::vector<Eigen::VectorXd> samples;
    std::vector<double> observations;
    std::vector<double> _noise;

    double sigma = 1, length = 1, noise = 1;
    double maximumObservedExpectedSquaredJumpDistance = 0;
    size_t maxElementIndex;
    Kernel kernel(sigma, length);
    GP gp = GP(kernel);

	for (size_t i = 0; i < parameters.maximumTotalIterations; ++i) {
        // train and GP
        gp = GP(kernel);
        gp.addObservations(samples, observations, _noise);
        
        // sample the acquisition function and obtain its maximum
        gp.sample(logStepSizeGrid, randomNumberGenerator[0], maxElementIndex);
        Eigen::VectorXd testStepSize = logStepSizeGrid[maxElementIndex];

        // evaluate stepsize which maximized the sampled acquisition function
        double unscaleFactor = maximumObservedExpectedSquaredJumpDistance;
        auto evaluations = measureExpectedSquaredJumpDistance(std::pow(10, testStepSize(0)), markovChain, randomNumberGenerator, parameters);
        for (size_t j = 0; j < evaluations.size(); ++j) {
            samples.push_back(testStepSize);
            observations.push_back(evaluations[j]);
            _noise.push_back(noise);

            // update max observed evaluation
            if (evaluations[j] > maximumObservedExpectedSquaredJumpDistance) {
                maximumObservedExpectedSquaredJumpDistance = evaluations[j];
            }
        }

        // the observations which were already recorded have to be rescaled
        for (size_t j = 0; j < observations.size() - evaluations.size(); ++j) {
            // if the unscaleFactor is zero, then because all previous observations
            // were zero, so they may as well be multiplied with zero
            observations[j] *= unscaleFactor;  
            // if the new maximum is zero, then also the old was, so no rescaling is done (=division by one)
            observations[j] /= (maximumObservedExpectedSquaredJumpDistance != 0 ? maximumObservedExpectedSquaredJumpDistance : 1);
        }

        // the new observations have not yet been scaled
        for (size_t j = observations.size() - evaluations.size(); j < observations.size(); ++j) {
            // if the new maximum is zero, then also the old was, so no rescaling is done (=division by one)
            observations[j] /= (maximumObservedExpectedSquaredJumpDistance != 0 ? maximumObservedExpectedSquaredJumpDistance : 1);
        }

        if (i == parameters.maximumTotalIterations - 1) {
            // unscale the posterior mean such that it has "correct" magnitude
            maximumExpectedSquaredJumpDistance = gp.getPosteriorMean().maxCoeff(&maxElementIndex) * maximumObservedExpectedSquaredJumpDistance;
            stepSize = std::pow(10, (logStepSizeGrid[maxElementIndex](0)));
        }
    }

    // just for debug and test purposes
    // for (size_t i = 0; i < samples.size(); ++i) {
    //     std::cout << samples[i] << " " << observations[i] << ";" << std::endl;
    // }
    // 
    // for (size_t i = 0; i < gp.getPosteriorMean().size(); ++i) {
    //     std::cout << logStepSizeGrid[i] << " " << gp.getPosteriorMean()(i) << " " << gp.getPosteriorCovariance()(i,i) <<  ";" << std::endl;
    // }
    
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
                                                               bool considerTimeCost) {
    this->iterationsToTestStepSize = iterationsToTestStepSize;
    this->maximumTotalIterations = maximumTotalIterations;
    this->stepSizeGridSize = stepSizeGridSize;
    this->stepSizeLowerBound = stepSizeLowerBound;
    this->stepSizeUpperBound = stepSizeUpperBound;
    this->considerTimeCost = considerTimeCost;
}

