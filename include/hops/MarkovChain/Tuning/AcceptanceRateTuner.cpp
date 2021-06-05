#include <hops/MarkovChain/MarkovChain.hpp>
#include <hops/MarkovChain/MarkovChainAttribute.hpp>
#include <hops/MarkovChain/Tuning/AcceptanceRateTuner.hpp>
#include <hops/Optimization/GaussianProcess.hpp>
#include <hops/Optimization/ThompsonSampling.hpp>

#include <cmath>
#include <memory>
#include <chrono>

/**
 * @brief measures the stepsize of a configured step size
 * @param stepSize
 * @param markovChain
 * @return
 */
std::vector<double> hops::internal::AcceptanceRateTarget::operator()(const Eigen::VectorXd& x) {
    double stepSize = std::pow(10, x(0));
    std::vector<double> acceptanceRateScores(markovChain.size());
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

        double acceptanceRate = markovChain[i]->getAcceptanceRate();
        std::cout << std::log10(stepSize) << ", " << acceptanceRate << std::endl;
        acceptanceRateScores[i] = 1 - std::abs(acceptanceRate - parameters.acceptanceRateTargetValue);
    }

    return acceptanceRateScores;
}

bool hops::AcceptanceRateTuner::tune(
        double& stepSize,
        double& deltaAcceptanceRate,
        std::vector<std::shared_ptr<hops::MarkovChain>>& markovChain,
        std::vector<RandomNumberGenerator>& randomNumberGenerator,
        const hops::AcceptanceRateTuner::param_type& parameters) {
    using Kernel = SquaredExponentialKernel<Eigen::MatrixXd, Eigen::VectorXd>;
    using GP = GaussianProcess<Eigen::MatrixXd, Eigen::VectorXd, Kernel>;

    std::vector<Eigen::VectorXd> logStepSizeGrid;
    double a = std::log10(parameters.stepSizeLowerBound), b = std::log10(parameters.stepSizeUpperBound);

    for (size_t i = 0; i < parameters.stepSizeGridSize; ++i) {
        Eigen::VectorXd x = Eigen::VectorXd(1);
        x(0) = (b - a) * i / (parameters.stepSizeGridSize - 1) + a;
        logStepSizeGrid.push_back(x);
    }

    double sigma = 0.5, length = 0.5, noise = 0.1;
    Kernel kernel(sigma, length);
    GP gp = GP(kernel);

    auto target = std::make_shared<internal::AcceptanceRateTarget>(internal::AcceptanceRateTarget(markovChain, randomNumberGenerator, parameters));

    std::vector<Eigen::VectorXd> samples;
    std::vector<double> observations;

    RandomNumberGenerator thompsonSamplingRandomNumberGenerator(parameters.randomSeed, markovChain.size() + 1);
    bool success = ThompsonSampling<Eigen::MatrixXd, Eigen::VectorXd, GP>::optimize(
            parameters.maximumTotalIterations,
            gp, target, logStepSizeGrid, 
            thompsonSamplingRandomNumberGenerator,
            samples, observations, 
            noise);
   
    auto& posteriorMean = gp.getPosteriorMean();
    auto& posteriorCovariance = gp.getPosteriorCovariance();

    size_t maximumIndex = 0;
    for (size_t i = 1; i < posteriorMean.size(); ++i) {
        if (posteriorMean(i) > posteriorMean(maximumIndex)) {
            maximumIndex = i;
        }
    }

    std::cout << "posterior = np.array([";
    for (size_t i = 0; i < posteriorMean.size(); ++i) {
        std::cout << "[" << logStepSizeGrid[i](0) << ", " << posteriorMean(i) << ", " << posteriorCovariance(i,i) << "], ";
    }
    std::cout << "])" << std::endl;

    std::cout << "data = np.array([";
    for (size_t i = 0; i < samples.size(); ++i) {
        std::cout << "[" << samples[i](0) << ", " << observations[i] << "], ";
    }
    std::cout << "])" << std::endl;

    stepSize = std::pow(10, logStepSizeGrid[maximumIndex](0));
    double maximumScore = posteriorMean(maximumIndex);

    deltaAcceptanceRate = 1 - maximumScore;

    return true;
}

bool hops::AcceptanceRateTuner::tune(
        std::vector<std::shared_ptr<hops::MarkovChain>>& markovChain,
        std::vector<RandomNumberGenerator>& randomNumberGenerator,
        const hops::AcceptanceRateTuner::param_type& parameters) {
    double stepSize = markovChain[0]->getAttribute(hops::MarkovChainAttribute::STEP_SIZE);
    double maximumAcceptanceRate;
    return tune(stepSize, maximumAcceptanceRate, markovChain, randomNumberGenerator, parameters);
}

hops::AcceptanceRateTuner::param_type::param_type(double acceptanceRateTargetValue,
                                                  size_t iterationsToTestStepSize,
                                                  size_t maximumTotalIterations,
                                                  size_t stepSizeGridSize,
                                                  double stepSizeLowerBound,
                                                  double stepSizeUpperBound,
                                                  size_t randomSeed) {
    this->acceptanceRateTargetValue = acceptanceRateTargetValue;
    this->iterationsToTestStepSize = iterationsToTestStepSize;
    this->maximumTotalIterations = maximumTotalIterations;
    this->stepSizeGridSize = stepSizeGridSize;
    this->stepSizeLowerBound = stepSizeLowerBound;
    this->stepSizeUpperBound = stepSizeUpperBound;
    this->randomSeed = randomSeed;
}

