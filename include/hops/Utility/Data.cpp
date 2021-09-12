#include "Data.hpp" 

Eigen::VectorXd hops::computeAcceptanceRate(hops::Data& data) {
    data.acceptanceRate = Eigen::VectorXd(data.chains.size());
    for (size_t i = 0; i < data.chains.size(); ++i) {
        data.acceptanceRate(i) = data.chains[i].getAcceptanceRates().back();
    }
    return data.acceptanceRate;
}

Eigen::VectorXd hops::computeEffectiveSampleSize(hops::Data& data) {
    std::vector<const std::vector<Eigen::VectorXd>*> states(data.chains.size());
    if (!data.chains.size()) {
        throw EmptyChainDataException();
    }

    for (size_t i = 0; i < states.size(); ++i) {
        states[i] = data.chains[i].states.get();
        if (!states[i]) {
            throw EmptyChainDataException();
        }
        if (i == 0 && states[i]->size() > 0) {
            data.dimension = states[i]->at(0).size();

        }
    }
    std::vector<double> effectiveSampleSize = ::hops::computeEffectiveSampleSize(states);
    data.effectiveSampleSize = Eigen::Map<Eigen::VectorXd>(effectiveSampleSize.data(), data.dimension);

    return data.effectiveSampleSize;
}

Eigen::VectorXd hops::computeExpectedSquaredJumpDistance(const hops::Data& data, const Eigen::MatrixXd& sqrtCovariance) {
    std::vector<const std::vector<Eigen::VectorXd>*> states(data.chains.size());
    if (!data.chains.size()) {
        throw EmptyChainDataException();
    }

    // collect states pointers
    for (size_t i = 0; i < states.size(); ++i) {
        states[i] = data.chains[i].states.get();
        if (!states[i]) {
            throw EmptyChainDataException();
        }
    }

    if (!sqrtCovariance.size()) {
        return computeExpectedSquaredJumpDistance<Eigen::VectorXd, Eigen::MatrixXd>(states);
    } else {
        return computeExpectedSquaredJumpDistance<Eigen::VectorXd, Eigen::MatrixXd>(states, sqrtCovariance);
    }
}

Eigen::VectorXd hops::computeExpectedSquaredJumpDistance(const hops::Data& data) {
    Eigen::MatrixXd dummySqrtCovariance(0, 0);
    return computeExpectedSquaredJumpDistance(data, dummySqrtCovariance);
}

Eigen::VectorXd hops::computePotentialScaleReductionFactor(hops::Data& data) {
    std::vector<const std::vector<Eigen::VectorXd>*> states(data.chains.size());
    if (!data.chains.size()) {
        throw EmptyChainDataException();
    }

    for (size_t i = 0; i < states.size(); ++i) {
        states[i] = data.chains[i].states.get();
        if (!states[i]) {
            throw EmptyChainDataException();
        }
        if (i == 0 && states[i]->size() > 0) {
            data.dimension = states[i]->at(0).size();

        }
    }
    std::vector<double> potentialScaleReductionFactor = ::hops::computePotentialScaleReductionFactor(states);
    data.potentialScaleReductionFactor = Eigen::Map<Eigen::VectorXd>(potentialScaleReductionFactor.data(), data.dimension);

    return data.potentialScaleReductionFactor;
}

Eigen::VectorXd hops::computeTotalTimeTaken(hops::Data& data) {
    data.totalTimeTaken = Eigen::VectorXd(data.chains.size());
    for (size_t i = 0; i < data.chains.size(); ++i) {
        data.totalTimeTaken(i) = data.chains[i].getTimestamps().back() - data.chains[i].getTimestamps().front();
    }
    return data.totalTimeTaken;
}

long computeTotalNumberOfSamples(hops::Data& data) {
    data.totalNumberOfSamples = 0;
    for (size_t i = 0; i < data.chains.size(); ++i) {
        data.totalNumberOfSamples += data.chains[i].getStates().size();
    }

    return data.totalNumberOfSamples;
}


std::tuple<Eigen::VectorXd, hops::IntermediateExpectedSquaredJumpDistanceResults_>
hops::computeExpectedSquaredJumpDistanceIncrementally(const hops::Data& data, const Eigen::MatrixXd& sqrtCovariance) {
    std::vector<const std::vector<Eigen::VectorXd>*> states(data.chains.size());
    if (!data.chains.size()) {
        throw EmptyChainDataException();
    }

    // collect states pointers
    for (size_t i = 0; i < states.size(); ++i) {
        states[i] = data.chains[i].states.get();
        if (!states[i]) {
            throw EmptyChainDataException();
        }
    }

    if (!sqrtCovariance.size()) {
        return computeExpectedSquaredJumpDistanceIncrementally<Eigen::VectorXd, Eigen::MatrixXd>(states);
    } else {
        return computeExpectedSquaredJumpDistanceIncrementally<Eigen::VectorXd, Eigen::MatrixXd>(states, sqrtCovariance);
    }
}

std::tuple<Eigen::VectorXd, hops::IntermediateExpectedSquaredJumpDistanceResults_>
hops::computeExpectedSquaredJumpDistanceIncrementally(const hops::Data& data) {
    Eigen::MatrixXd dummySqrtCovariance(0, 0);
    return computeExpectedSquaredJumpDistanceIncrementally(data, dummySqrtCovariance);
}

std::tuple<Eigen::VectorXd, hops::IntermediateExpectedSquaredJumpDistanceResults_>
hops::computeExpectedSquaredJumpDistanceIncrementally(const hops::Data& data, 
                                                      const hops::IntermediateExpectedSquaredJumpDistanceResults_& intermediateResults, 
                                                      const Eigen::MatrixXd& sqrtCovariance) {
    std::vector<const std::vector<Eigen::VectorXd>*> states(data.chains.size());
    if (!data.chains.size()) {
        throw EmptyChainDataException();
    }

    // collect states pointers
    for (size_t i = 0; i < states.size(); ++i) {
        states[i] = data.chains[i].states.get();
        if (!states[i]) {
            throw EmptyChainDataException();
        }
    }

    if (!sqrtCovariance.size()) {
        return ::hops::computeExpectedSquaredJumpDistanceIncrementally<Eigen::VectorXd, Eigen::MatrixXd>(states, intermediateResults);
    } else {
        return ::hops::computeExpectedSquaredJumpDistanceIncrementally<Eigen::VectorXd, Eigen::MatrixXd>(states, sqrtCovariance, intermediateResults);
    }
}

std::tuple<Eigen::VectorXd, hops::IntermediateExpectedSquaredJumpDistanceResults_>
hops::computeExpectedSquaredJumpDistanceIncrementally(const hops::Data& data, 
                                                      const hops::IntermediateExpectedSquaredJumpDistanceResults_& intermediateResults) {
    Eigen::MatrixXd dummySqrtCovariance(0, 0);
    return computeExpectedSquaredJumpDistanceIncrementally(data, intermediateResults, dummySqrtCovariance);
}

Eigen::MatrixXd hops::computeExpectedSquaredJumpDistanceEvery(const Data& data, size_t k, const Eigen::MatrixXd& sqrtCovariance) {
    std::vector<const std::vector<Eigen::VectorXd>*> states(data.chains.size());
    if (!data.chains.size()) {
        throw EmptyChainDataException();
    }

    // collect states pointers
    for (size_t i = 0; i < states.size(); ++i) {
        states[i] = data.chains[i].states.get();
        if (!states[i]) {
            throw EmptyChainDataException();
        }
    }

    if (!sqrtCovariance.size()) {
        return ::hops::computeExpectedSquaredJumpDistanceEveryKth<Eigen::VectorXd, Eigen::MatrixXd>(states, k);
    } else {
        return ::hops::computeExpectedSquaredJumpDistanceEveryKth<Eigen::VectorXd, Eigen::MatrixXd>(states, k, sqrtCovariance);
    }
}

Eigen::MatrixXd hops::computeExpectedSquaredJumpDistanceEvery(const Data& data, size_t k) {
    Eigen::MatrixXd dummySqrtCovariance(0, 0);
    return computeExpectedSquaredJumpDistanceEvery(data, k, dummySqrtCovariance);
}

