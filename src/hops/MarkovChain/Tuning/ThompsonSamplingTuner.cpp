#include "ThompsonSamplingTuner.hpp"

hops::ThompsonSamplingTuner::param_type::param_type(size_t posteriorUpdateIterations,
                                                    size_t pureSamplingIterations,
                                                    size_t iterationsForConvergence,
                                                    size_t stepSizeGridSize,
                                                    double stepSizeLowerBound,
                                                    double stepSizeUpperBound,
                                                    double smoothingLength,
                                                    size_t randomSeed,
                                                    bool recordData) {
    this->posteriorUpdateIterations = posteriorUpdateIterations;
    this->pureSamplingIterations = pureSamplingIterations;
    this->iterationsForConvergence = iterationsForConvergence;
    this->posteriorUpdateIterationsNeeded = 0;
    this->stepSizeGridSize = stepSizeGridSize;
    this->stepSizeLowerBound = stepSizeLowerBound;
    this->stepSizeUpperBound = stepSizeUpperBound;
    this->smoothingLength = smoothingLength;
    this->randomSeed = randomSeed;
    this->recordData = recordData;
}

