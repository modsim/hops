#include <cmath>

#include "BinarySearchAcceptanceRateTuner.hpp"
#include <vector>

namespace {
    enum class Case {
        ACCEPTANCE_RATE_TOO_LOW,
        ACCEPTANCE_RATE_TOO_HIGH,
        ACCEPTANCE_RATE_GOOD
    };
}

Case currentCase(double measuredAcceptanceRate, const hops::BinarySearchAcceptanceRateTuner::param_type &parameters) {
    if (measuredAcceptanceRate < parameters.lowerLimitAcceptanceRate) {
        return Case::ACCEPTANCE_RATE_TOO_LOW;
    } else if (measuredAcceptanceRate > parameters.upperLimitAcceptanceRate) {
        return Case::ACCEPTANCE_RATE_TOO_HIGH;
    } else {
        return Case::ACCEPTANCE_RATE_GOOD;
    }
}

/**
 * @brief measures the acceptance rate of a configured step size
 * @param stepSize
 * @param markovChain
 * @return
 */
double measureAcceptanceRate(double stepSize,
                             hops::MarkovChain *markovChain,
                             hops::RandomNumberGenerator randomNumberGenerator,
                             const hops::BinarySearchAcceptanceRateTuner::param_type &parameters) {
    markovChain->setParameter(hops::ProposalParameter::STEP_SIZE, stepSize);
    auto[acceptanceRate, _] = markovChain->draw(randomNumberGenerator, parameters.iterationsToTestStepSize);
    return acceptanceRate;
}

/**
 * @brief determines which step size to try next based on the measured acceptance rate of the current step size
 * @param currentStepSize
 * @param measuredAcceptanceRate
 * @return
 */
double nextStepSizeToTry(double currentStepSize,
                         double measuredAcceptanceRate,
                         const hops::BinarySearchAcceptanceRateTuner::param_type &parameters) {
    switch (currentCase(measuredAcceptanceRate, parameters)) {
        case Case::ACCEPTANCE_RATE_TOO_HIGH:
            if (currentStepSize > parameters.lowerLimitStepSize) {
                parameters.lowerLimitStepSize = currentStepSize;
            }
            return parameters.upperLimitStepSize == std::numeric_limits<double>::infinity() ?
                   currentStepSize * 2 :
                   (currentStepSize + parameters.upperLimitStepSize) / 2;
        case Case::ACCEPTANCE_RATE_TOO_LOW:
            if (currentStepSize < parameters.upperLimitStepSize) {
                parameters.upperLimitStepSize = currentStepSize;
            }
            return (currentStepSize + parameters.lowerLimitStepSize) / 2;
        default:
            return currentStepSize;
    }
}

bool hops::BinarySearchAcceptanceRateTuner::tune(MarkovChain *markovChain,
                                                 RandomNumberGenerator &randomNumberGenerator,
                                                 const hops::BinarySearchAcceptanceRateTuner::param_type &parameters) {
    double stepSize = std::any_cast<double>(markovChain->getParameter(ProposalParameter::STEP_SIZE));
    double acceptanceRate;
    return tune(stepSize, acceptanceRate, markovChain, randomNumberGenerator, parameters);
}


bool hops::BinarySearchAcceptanceRateTuner::tune(double &stepSize,
                                                 double &acceptanceRate,
                                                 MarkovChain *markovChain,
                                                 RandomNumberGenerator &randomNumberGenerator,
                                                 const hops::BinarySearchAcceptanceRateTuner::param_type &parameters) {
    double currentAcceptanceRate = 0;
    size_t iterationsCount = 0;
    while (currentCase(currentAcceptanceRate, parameters) != Case::ACCEPTANCE_RATE_GOOD) {
        //markovChain->clearHistory();
        if (iterationsCount > parameters.maximumTotalIterations) {
            return false;
        }
        currentAcceptanceRate = measureAcceptanceRate(stepSize, markovChain, randomNumberGenerator, parameters);
        iterationsCount += parameters.iterationsToTestStepSize;
        stepSize = nextStepSizeToTry(stepSize, currentAcceptanceRate, parameters);
        markovChain->setParameter(ProposalParameter::STEP_SIZE, stepSize);
    }
    acceptanceRate = currentAcceptanceRate;
    return true;
}


/**
 * @brief tunes markov chain acceptance rate by nested intervals. The chain is not guaranteed to have converged
 *        to the specified acceptance rate.
 * @details Clears Markov chain history.
 * @param markovChain
 * @param parameters
 * @return true if markov chain is tuned
 */
bool
hops::BinarySearchAcceptanceRateTuner::tune(std::vector<std::shared_ptr<hops::MarkovChain>> &markovChain,
                                            std::vector<hops::RandomNumberGenerator> &randomNumberGenerator,
                                            const param_type &parameters) {
    double stepSize, acceptanceRate;
    return tune(stepSize, acceptanceRate, markovChain, randomNumberGenerator, parameters);
}

/**
 * @brief tunes markov chain acceptance rate by nested intervals. The chain is not guaranteed to have converged
 *        to the specified acceptance rate.
 * @details Clears Markov chain history.
 * @param markovChain
 * @param parameters
 * @return true if markov chain is tuned
 */
bool
hops::BinarySearchAcceptanceRateTuner::tune(double &stepSize,
                                            double &acceptanceRate,
                                            std::vector<std::shared_ptr<hops::MarkovChain>> &markovChain,
                                            std::vector<hops::RandomNumberGenerator> &randomNumberGenerator,
                                            const param_type &parameters) {
    bool tuned = tune(stepSize, acceptanceRate, markovChain[0].get(), randomNumberGenerator[0], parameters);
    for (auto &chain : markovChain) {
        chain->setParameter(ProposalParameter::STEP_SIZE, stepSize);
    }
    return tuned;
}


hops::BinarySearchAcceptanceRateTuner::param_type::param_type(double lowerLimitAcceptanceRate, double upperLimitAcceptanceRate,
                                                              double lowerLimitStepSize, double upperLimitStepSize,
                                                              size_t iterationsToTestStepSize,
                                                              size_t maximumTotalIterations) {
    if (lowerLimitAcceptanceRate >= upperLimitAcceptanceRate) {
        throw std::runtime_error("Parameter error: lowerLimitAcceptanceRate is larger than upperLimitAcceptanceRate");
    }
    if (lowerLimitStepSize >= upperLimitStepSize) {
        throw std::runtime_error("Parameter error: lowerLimitStepSize is larger than upperLimitStepSize");
    }
    if (iterationsToTestStepSize == 0) {
        throw std::runtime_error("Parameter error: iterationsToTestStepSize is 0");
    }

    this->lowerLimitAcceptanceRate = lowerLimitAcceptanceRate;
    this->upperLimitAcceptanceRate = upperLimitAcceptanceRate;
    this->lowerLimitStepSize = lowerLimitStepSize;
    this->upperLimitStepSize = upperLimitStepSize;
    this->iterationsToTestStepSize = iterationsToTestStepSize;
    this->maximumTotalIterations = maximumTotalIterations;
}
