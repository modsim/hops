#include "AcceptanceRateTuner.hpp"

/**
 * @brief measures the stepsize of a configured step size
 * @param stepSize
 * @param markovChains
 * @return
 */
//std::tuple<double, double> hops::internal::AcceptanceRateTarget::operator()(const Eigen::VectorXd& x) {
//    double stepSize = std::pow(10, x(0));
//    std::vector<double> acceptanceRateScores(markovChains.size());
//    #pragma omp parallel for num_threads(numberOfThreads)
//    for (size_t i = 0; i < markovChains.size(); ++i) {
//        //markovChains[i]->clearHistory();
//        //markovChains[i]->setAttribute(hops::MarkovChainAttribute::STEP_SIZE, stepSize);
//        markovChains[i]->setParameter(ProposalParameter::STEP_SIZE, stepSize);
//
//        // record time taken to draw samples to scale esjd by time if specified
//        unsigned long time = std::chrono::duration_cast<std::chrono::milliseconds>(
//                std::chrono::high_resolution_clock::now().time_since_epoch()
//        ).count();
//
//        markovChains[i]->setParameter(hops::ProposalParameter::STEP_SIZE, stepSize);
//        auto[acceptanceRate, _] = markovChains[i]->draw(randomNumberGenerator->at(i), parameters.iterationsToTestStepSize);
//
//        time = std::chrono::duration_cast<std::chrono::milliseconds>(
//                std::chrono::high_resolution_clock::now().time_since_epoch()
//        ).count() - time;
//
//        // set time to 1 if it was 0
//        time = (time == 0 ? 1 : time);
//
//        double deltaScale = (
//                acceptanceRate > parameters.acceptanceRateTargetValue ?
//                1 - parameters.acceptanceRateTargetValue :
//                parameters.acceptanceRateTargetValue
//        );
//        acceptanceRateScores[i] = 1 - std::abs(acceptanceRate - parameters.acceptanceRateTargetValue) / deltaScale;
//    }
//
//    double mean = std::accumulate(acceptanceRateScores.begin(), acceptanceRateScores.end(), 0.0) / acceptanceRateScores.size();
//
//    double squaredSum = std::inner_product(acceptanceRateScores.begin(), acceptanceRateScores.end(), acceptanceRateScores.begin(), 0.0);
//    //double error = std::sqrt(squaredSum / acceptanceRateScores.size() - mean * mean); 
//    double error = squaredSum / acceptanceRateScores.size() - mean * mean; 
//
//    return {mean, error};
//}

bool hops::AcceptanceRateTuner::tune(
        VectorType& stepSize,
        double& deltaAcceptanceRate,
        std::vector<std::shared_ptr<hops::MarkovChain>>& markovChains,
        const std::vector<RandomNumberGenerator*>& randomNumberGenerators,
        AcceptanceRateTuner::param_type &parameters,
        Eigen::MatrixXd& data) {
    auto target = AcceptanceRateTarget(markovChains, parameters.iterationsToTestStepSize, parameters.acceptanceRateTargetValue);
    return ThompsonSamplingTuner::tune(stepSize, deltaAcceptanceRate, randomNumberGenerators, parameters.ts_params, target, data);

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

    //auto target = AcceptanceRateTarget(markovChains, parameters.iterationsToTestStepSize, parameters.acceptanceRateTargetValue);

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

    //auto& posteriorMean = gp.getPosteriorMean();
    //size_t maximumIndex;
    //double maximumScore = posteriorMean.maxCoeff(&maximumIndex);
    //stepSize = std::pow(10, logStepSizeGrid(maximumIndex, 0));

    //deltaAcceptanceRate = 1 - maximumScore;

    //return isThompsonSamplingConverged;
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
    this->ts_params.iterationsToTestStepSize = iterationsToTestStepSize;
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

