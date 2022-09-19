#ifndef HOPS_EXPECTEDSQUAREDJUMPDISTANCE_HPP
#define HOPS_EXPECTEDSQUAREDJUMPDISTANCE_HPP

#include <cassert>
#include <cmath>
#include <Eigen/Core>
#include <Eigen/Cholesky>
#include <limits>
#include <memory>
#include <string>
#include <stdexcept>
#include <tuple>
#include <vector>

#include "hops/Statistics/Covariance.hpp"
#include "hops/Statistics/IsConstantChain.hpp"


namespace hops {
    // intermediate results needed to compute expected squared jump distance incrementally are:
    //      vector of the esjds of the states already seen,
    //      number of states already seen,
    //      intermediate covariance results to compute covariance incrementally
    template<typename StateType, typename MatrixType>
    using IntermediateExpectedSquaredJumpDistanceResults = std::tuple<StateType, unsigned long, IntermediateCovarianceResults<StateType, MatrixType>>;



    template<typename StateType, typename MatrixType>
    std::tuple<StateType, IntermediateExpectedSquaredJumpDistanceResults<StateType, MatrixType>>
    computeExpectedSquaredJumpDistanceIncrementally(const std::vector<const std::vector<StateType>*>& chains, 
                                                    const MatrixType& sqrtCovariance,
                                                    const IntermediateExpectedSquaredJumpDistanceResults<StateType, MatrixType>& intermediateResults,
                                                    size_t lag = 1, size_t start = 0, size_t stop = std::numeric_limits<size_t>::max(), size_t step = 1);

    template<typename StateType, typename MatrixType>
    std::tuple<StateType, IntermediateExpectedSquaredJumpDistanceResults<StateType, MatrixType>>
    computeExpectedSquaredJumpDistanceIncrementally(const std::vector<const std::vector<StateType>*>& chains, 
                                                    const MatrixType& sqrtCovariance,
                                                    size_t lag = 1, size_t start = 0, size_t stop = std::numeric_limits<size_t>::max(), size_t step = 1);

    template<typename StateType, typename MatrixType>
    StateType computeExpectedSquaredJumpDistance(const std::vector<const std::vector<StateType>*>& chains, 
                                                 const MatrixType& sqrtCovariance,
                                                 size_t lag = 1, size_t start = 0, size_t stop = std::numeric_limits<size_t>::max(), size_t step = 1);

    template<typename StateType, typename MatrixType>
    MatrixType computeExpectedSquaredJumpDistanceEveryKth(const std::vector<const std::vector<StateType>*>& chains, size_t k,
                                                          const MatrixType& sqrtCovariance,
                                                          size_t lag = 1, size_t start = 0, size_t stop = std::numeric_limits<size_t>::max(), size_t step = 1);




    template<typename StateType, typename MatrixType>
    std::tuple<StateType, IntermediateExpectedSquaredJumpDistanceResults<StateType, MatrixType>>
    computeExpectedSquaredJumpDistanceIncrementally(const std::vector<const std::vector<StateType>*>& chains, 
                                                    const IntermediateExpectedSquaredJumpDistanceResults<StateType, MatrixType>& intermediateResults,
                                                    size_t lag = 1, size_t start = 0, size_t stop = std::numeric_limits<size_t>::max(), size_t step = 1);

    template<typename StateType, typename MatrixType>
    std::tuple<StateType, IntermediateExpectedSquaredJumpDistanceResults<StateType, MatrixType>>
    computeExpectedSquaredJumpDistanceIncrementally(const std::vector<const std::vector<StateType>*>& chains, 
                                                    size_t lag = 1, size_t start = 0, size_t stop = std::numeric_limits<size_t>::max(), size_t step = 1);

    template<typename StateType, typename MatrixType>
    StateType computeExpectedSquaredJumpDistance(const std::vector<const std::vector<StateType>*>& chains, 
                                                 size_t lag = 1, size_t start = 0, size_t stop = std::numeric_limits<size_t>::max(), size_t step = 1);

    template<typename StateType, typename MatrixType>
    MatrixType computeExpectedSquaredJumpDistanceEveryKth(const std::vector<const std::vector<StateType>*>& chains, size_t k, 
                                                          size_t lag = 1, size_t start = 0, size_t stop = std::numeric_limits<size_t>::max(), size_t step = 1);




    template<typename StateType, typename MatrixType>
    std::tuple<StateType, IntermediateExpectedSquaredJumpDistanceResults<StateType, MatrixType>>
    computeExpectedSquaredJumpDistanceIncrementally(const std::vector<std::vector<StateType>>& chains, 
                                                    const MatrixType& sqrtCovariance,
                                                    const IntermediateExpectedSquaredJumpDistanceResults<StateType, MatrixType>& intermediateResults,
                                                    size_t lag = 1, size_t start = 0, size_t stop = std::numeric_limits<size_t>::max(), size_t step = 1);

    template<typename StateType, typename MatrixType>
    std::tuple<StateType, IntermediateExpectedSquaredJumpDistanceResults<StateType, MatrixType>>
    computeExpectedSquaredJumpDistanceIncrementally(const std::vector<std::vector<StateType>>& chains,
                                                    const MatrixType& sqrtCovariance,
                                                    size_t lag = 1, size_t start = 0, size_t stop = std::numeric_limits<size_t>::max(), size_t step = 1);

    template<typename StateType, typename MatrixType>
    StateType computeExpectedSquaredJumpDistance(const std::vector<std::vector<StateType>>& chains,
                                                 const MatrixType& sqrtCovariance,
                                                 size_t lag = 1, size_t start = 0, size_t stop = std::numeric_limits<size_t>::max(), size_t step = 1);

    template<typename StateType, typename MatrixType>
    MatrixType computeExpectedSquaredJumpDistanceEveryKth(const std::vector<std::vector<StateType>>& chains, size_t k,
                                                          const MatrixType& sqrtCovariance,
                                                          size_t lag = 1, size_t start = 0, size_t stop = std::numeric_limits<size_t>::max(), size_t step = 1);



    template<typename StateType, typename MatrixType>
    std::tuple<StateType, IntermediateExpectedSquaredJumpDistanceResults<StateType, MatrixType>>
    computeExpectedSquaredJumpDistanceIncrementally(const std::vector<std::vector<StateType>>& chains, 
                                                    const IntermediateExpectedSquaredJumpDistanceResults<StateType, MatrixType>& intermediateResults,
                                                    size_t lag = 1, size_t start = 0, size_t stop = std::numeric_limits<size_t>::max(), size_t step = 1);

    template<typename StateType, typename MatrixType>
    std::tuple<StateType, IntermediateExpectedSquaredJumpDistanceResults<StateType, MatrixType>>
    computeExpectedSquaredJumpDistanceIncrementally(const std::vector<std::vector<StateType>>& chains,
                                                          size_t lag = 1, size_t start = 0, size_t stop = std::numeric_limits<size_t>::max(), size_t step = 1);

    template<typename StateType, typename MatrixType>
    StateType computeExpectedSquaredJumpDistance(const std::vector<std::vector<StateType>>& chains,
                                                          size_t lag = 1, size_t start = 0, size_t stop = std::numeric_limits<size_t>::max(), size_t step = 1);

    template<typename StateType, typename MatrixType>
    MatrixType computeExpectedSquaredJumpDistanceEveryKth(const std::vector<std::vector<StateType>>& chains, size_t k,
                                                          size_t lag = 1, size_t start = 0, size_t stop = std::numeric_limits<size_t>::max(), size_t step = 1);



    template<typename StateType, typename MatrixType>
    std::tuple<double, IntermediateExpectedSquaredJumpDistanceResults<StateType, MatrixType>>
    computeExpectedSquaredJumpDistanceIncrementally(const std::vector<StateType>& chains, 
                                                    const MatrixType& sqrtCovariance,
                                                    const IntermediateExpectedSquaredJumpDistanceResults<StateType, MatrixType>& intermediateResults,
                                                    size_t lag = 1, size_t start = 0, size_t stop = std::numeric_limits<size_t>::max(), size_t step = 1);

    template<typename StateType, typename MatrixType>
    std::tuple<double, IntermediateExpectedSquaredJumpDistanceResults<StateType, MatrixType>>
    computeExpectedSquaredJumpDistanceIncrementally(const std::vector<StateType>& chains,
                                                    const MatrixType& sqrtCovariance,
                                                    size_t lag = 1, size_t start = 0, size_t stop = std::numeric_limits<size_t>::max(), size_t step = 1);

    template<typename StateType, typename MatrixType>
    double computeExpectedSquaredJumpDistance(const std::vector<StateType>& chains,
                                              const MatrixType& sqrtCovariance,
                                              size_t lag = 1, size_t start = 0, size_t stop = std::numeric_limits<size_t>::max(), size_t step = 1);

    template<typename StateType, typename MatrixType>
    StateType computeExpectedSquaredJumpDistanceEveryKth(const std::vector<StateType>& chains, size_t k,
                                                         const MatrixType& sqrtCovariance,
                                                         size_t lag = 1, size_t start = 0, size_t stop = std::numeric_limits<size_t>::max(), size_t step = 1);



    template<typename StateType, typename MatrixType>
    std::tuple<double, IntermediateExpectedSquaredJumpDistanceResults<StateType, MatrixType>>
    computeExpectedSquaredJumpDistanceIncrementally(const std::vector<StateType>& chains, 
                                                    const IntermediateExpectedSquaredJumpDistanceResults<StateType, MatrixType>& intermediateResults,
                                                    size_t lag = 1, size_t start = 0, size_t stop = std::numeric_limits<size_t>::max(), size_t step = 1);

    template<typename StateType, typename MatrixType>
    std::tuple<double, IntermediateExpectedSquaredJumpDistanceResults<StateType, MatrixType>>
    computeExpectedSquaredJumpDistanceIncrementally(const std::vector<StateType>& chains,
                                                    size_t lag = 1, size_t start = 0, size_t stop = std::numeric_limits<size_t>::max(), size_t step = 1);

    template<typename StateType, typename MatrixType>
    double computeExpectedSquaredJumpDistance(const std::vector<StateType>& chains,
                                              size_t lag = 1, size_t start = 0, size_t stop = std::numeric_limits<size_t>::max(), size_t step = 1);

    template<typename StateType, typename MatrixType>
    StateType computeExpectedSquaredJumpDistanceEveryKth(const std::vector<StateType>& chains, size_t k,
                                                         size_t lag = 1, size_t start = 0, size_t stop = std::numeric_limits<size_t>::max(), size_t step = 1);
}



template<typename StateType, typename MatrixType>
inline std::tuple<StateType, hops::IntermediateExpectedSquaredJumpDistanceResults<StateType, MatrixType>>
hops::computeExpectedSquaredJumpDistanceIncrementally(const std::vector<const std::vector<StateType>*>& chains, 
                                                      const MatrixType& sqrtCovariance,
                                                      const hops::IntermediateExpectedSquaredJumpDistanceResults<StateType, MatrixType>& intermediateResult,
                                                      size_t lag, size_t start, size_t stop, size_t step) {
        StateType expectedSquaredJumpDistances = StateType::Zero(chains.size());
        hops::IntermediateExpectedSquaredJumpDistanceResults<StateType, MatrixType> newIntermediateResult = intermediateResult;

        if (chains.size()) {
            // unpack intermediate results
            const auto&[intermediateExpectedSquaredJumpDistance, numberOfSeenStates, intermediateCovarianceResults] = intermediateResult;

            size_t numberOfStates, numberOfUnseenStates, correction, _start;

            for (size_t i = 0; i < chains.size(); ++i) {
                const std::vector<StateType>& states = *chains[i];

                numberOfStates = std::min(states.size(), stop);
                numberOfUnseenStates = numberOfStates - numberOfSeenStates;

                // the correction accounts for whether there have been states recorded previously
                // and for the "gap" esjd between two batches of states
                correction = ( numberOfSeenStates ? 0 : 1 );

                _start = std::max(numberOfSeenStates + correction - 1, start);
                double expectedSquaredJumpDistance = 0;

                // eta is the weight of the already seen expected squared jump distance
                double eta = 1.0 * (numberOfSeenStates + correction - 1) / (numberOfSeenStates + numberOfUnseenStates - 1);

                // a constant chain has an esjd of 0
                if (!isConstantChain<StateType>({states})) {
                    for (unsigned long j = _start; j < numberOfStates - lag; j += step) {
                        StateType distance = sqrtCovariance.template triangularView<Eigen::Lower>().solve(states[j] - states[j+lag]);
                        expectedSquaredJumpDistance += static_cast<typename StateType::Scalar>(distance.transpose() * distance);
                    }
                }

                expectedSquaredJumpDistance /= numberOfUnseenStates - correction;
                expectedSquaredJumpDistances(i) += (1. - eta) * expectedSquaredJumpDistance;
                if (eta) {
                    expectedSquaredJumpDistances(i) += eta * intermediateExpectedSquaredJumpDistance(i);
                }
            }

            newIntermediateResult = std::make_tuple(expectedSquaredJumpDistances, numberOfStates, intermediateCovarianceResults);
        }

        return {expectedSquaredJumpDistances, newIntermediateResult};
}

template<typename StateType, typename MatrixType>
inline std::tuple<StateType, hops::IntermediateExpectedSquaredJumpDistanceResults<StateType, MatrixType>>
hops::computeExpectedSquaredJumpDistanceIncrementally(const std::vector<const std::vector<StateType>*>& chains, 
                                                      const MatrixType& sqrtCovariance,
                                                      size_t lag, size_t start, size_t stop, size_t step) {
    hops::IntermediateExpectedSquaredJumpDistanceResults<StateType, MatrixType> zeroResult = std::make_tuple(
            StateType::Zero(chains.size()), 0, std::make_tuple(MatrixType::Identity(0, 0), StateType::Zero(0), 0));

    return hops::computeExpectedSquaredJumpDistanceIncrementally<StateType, MatrixType>(chains, sqrtCovariance, zeroResult, lag, start, stop, step);
}

template<typename StateType, typename MatrixType>
inline StateType hops::computeExpectedSquaredJumpDistance(const std::vector<const std::vector<StateType>*>& chains, 
                                                          const MatrixType& sqrtCovariance,
                                                          size_t lag, size_t start, size_t stop, size_t step) {
    hops::IntermediateExpectedSquaredJumpDistanceResults<StateType, MatrixType> zeroResult = std::make_tuple(
            StateType::Zero(chains.size()), 0, std::make_tuple(MatrixType::Identity(0, 0), StateType::Zero(0), 0));

    return std::get<0>(hops::computeExpectedSquaredJumpDistanceIncrementally<StateType, MatrixType>(chains, sqrtCovariance, zeroResult, lag, start, stop, step));
}

template<typename StateType, typename MatrixType>
inline MatrixType hops::computeExpectedSquaredJumpDistanceEveryKth(const std::vector<const std::vector<StateType>*>& chains, size_t k,  
                                                                   const MatrixType& sqrtCovariance,
                                                                   size_t lag, size_t start, size_t stop, size_t step) {
    size_t _stop = stop;
    if (chains.size()) {
        _stop = std::min(chains[0]->size(), stop); // here it is assumed, that all chains have the same length
    }

    size_t numberOfLogs = static_cast<size_t>(std::ceil(1. * (_stop - start) / k)),
           j = 0;

    MatrixType expectedSquaredJumpDistances = MatrixType::Zero(numberOfLogs, chains.size());
    hops::IntermediateExpectedSquaredJumpDistanceResults<StateType, MatrixType> intermediateResult = {
            StateType::Zero(chains.size()), 0, std::make_tuple(MatrixType::Identity(0, 0), StateType::Zero(0), 0)};

    for (size_t i = start + k; i <= _stop; i += k, ++j) {
            auto[expectedSquaredJumpDistance, _intermediateResult] = hops::computeExpectedSquaredJumpDistanceIncrementally<StateType, MatrixType>(chains, sqrtCovariance, intermediateResult, start, i, step, lag);
            intermediateResult = _intermediateResult;
            expectedSquaredJumpDistances.row(j) = expectedSquaredJumpDistance;
    }

    return expectedSquaredJumpDistances;
}



template<typename StateType, typename MatrixType>
inline std::tuple<StateType, hops::IntermediateExpectedSquaredJumpDistanceResults<StateType, MatrixType>>
hops::computeExpectedSquaredJumpDistanceIncrementally(const std::vector<const std::vector<StateType>*>& chains, 
                                                      const hops::IntermediateExpectedSquaredJumpDistanceResults<StateType, MatrixType>& intermediateResult,
                                                      size_t lag, size_t start, size_t stop, size_t step) {
    const auto&[intermediateExpectedSquaredJumpDistance, numberOfSeenStates, intermediateCovarianceResult] = intermediateResult;
    auto[covariance, newIntermediateCovarianceResult] = computeCovarianceIncrementally<StateType, MatrixType>(chains, intermediateCovarianceResult, start, stop, step);

    MatrixType sqrtCovariance = covariance.llt().matrixL();

    hops::IntermediateExpectedSquaredJumpDistanceResults<StateType, MatrixType> newIntermediateResult = 
        std::make_tuple(intermediateExpectedSquaredJumpDistance, numberOfSeenStates, newIntermediateCovarianceResult);
    return hops::computeExpectedSquaredJumpDistanceIncrementally(chains, sqrtCovariance, newIntermediateResult, lag, start, stop, step);
}

template<typename StateType, typename MatrixType>
inline std::tuple<StateType, hops::IntermediateExpectedSquaredJumpDistanceResults<StateType, MatrixType>>
hops::computeExpectedSquaredJumpDistanceIncrementally(const std::vector<const std::vector<StateType>*>& chains, 
                                                      size_t lag, size_t start, size_t stop, size_t step) {
    hops::IntermediateExpectedSquaredJumpDistanceResults<StateType, MatrixType> zeroResult = std::make_tuple(
            StateType::Zero(chains.size()), 0, std::make_tuple(MatrixType::Identity(0, 0), StateType::Zero(0), 0));

    return hops::computeExpectedSquaredJumpDistanceIncrementally<StateType, MatrixType>(chains, zeroResult, lag, start, stop, step);
}

template<typename StateType, typename MatrixType>
inline StateType hops::computeExpectedSquaredJumpDistance(const std::vector<const std::vector<StateType>*>& chains, 
                                                          size_t lag, size_t start, size_t stop, size_t step) {
    hops::IntermediateExpectedSquaredJumpDistanceResults<StateType, MatrixType> zeroResult = std::make_tuple(
            StateType::Zero(chains.size()), 0, std::make_tuple(MatrixType::Identity(0, 0), StateType::Zero(0), 0));

    return std::get<0>(hops::computeExpectedSquaredJumpDistanceIncrementally<StateType, MatrixType>(chains, zeroResult, lag, start, stop, step));
}

template<typename StateType, typename MatrixType>
inline MatrixType hops::computeExpectedSquaredJumpDistanceEveryKth(const std::vector<const std::vector<StateType>*>& chains, size_t k, 
                                                                   size_t lag, size_t start, size_t stop, size_t step) {
    auto[covariance, _] = computeCovarianceIncrementally<StateType, MatrixType>(chains, start, stop, step);
    MatrixType sqrtCovariance = covariance.llt().matrixL();
    return computeExpectedSquaredJumpDistanceEveryKth(chains, k, sqrtCovariance, lag, start, stop, step);
}



template<typename StateType, typename MatrixType>
inline std::tuple<StateType, hops::IntermediateExpectedSquaredJumpDistanceResults<StateType, MatrixType>>
hops::computeExpectedSquaredJumpDistanceIncrementally(const std::vector<std::vector<StateType>>& chains, 
                                                      const MatrixType& sqrtCovariance,
                                                      const hops::IntermediateExpectedSquaredJumpDistanceResults<StateType, MatrixType>& intermediateResult,
                                                      size_t lag, size_t start, size_t stop, size_t step) {
    std::vector<const std::vector<StateType>*> chainPtrArray;
    for (const auto& chain : chains) {
        chainPtrArray.push_back(&chain);
    }
    return hops::computeExpectedSquaredJumpDistanceIncrementally<StateType, MatrixType>(chainPtrArray, sqrtCovariance, intermediateResult, lag, start, stop, step);
}

template<typename StateType, typename MatrixType>
inline std::tuple<StateType, hops::IntermediateExpectedSquaredJumpDistanceResults<StateType, MatrixType>>
hops::computeExpectedSquaredJumpDistanceIncrementally(const std::vector<std::vector<StateType>>& chains,
                                                      const MatrixType& sqrtCovariance,
                                                      size_t lag, size_t start, size_t stop, size_t step) {
    std::vector<const std::vector<StateType>*> chainPtrArray;
    for (const auto& chain : chains) {
        chainPtrArray.push_back(&chain);
    }
    return hops::computeExpectedSquaredJumpDistanceIncrementally<StateType, MatrixType>(chainPtrArray, sqrtCovariance, lag, start, stop, step);
}

template<typename StateType, typename MatrixType>
inline StateType hops::computeExpectedSquaredJumpDistance(const std::vector<std::vector<StateType>>& chains,
                                                          const MatrixType& sqrtCovariance,
                                                          size_t lag, size_t start, size_t stop, size_t step) {
    std::vector<const std::vector<StateType>*> chainPtrArray;
    for (const auto& chain : chains) {
        chainPtrArray.push_back(&chain);
    }
    return hops::computeExpectedSquaredJumpDistance<StateType, MatrixType>(chainPtrArray, sqrtCovariance, lag, start, stop, step);
}

template<typename StateType, typename MatrixType>
inline MatrixType hops::computeExpectedSquaredJumpDistanceEveryKth(const std::vector<std::vector<StateType>>& chains, size_t k,
                                                                   const MatrixType& sqrtCovariance,
                                                                   size_t lag, size_t start, size_t stop, size_t step) {
    std::vector<const std::vector<StateType>*> chainPtrArray;
    for (const auto& chain : chains) {
        chainPtrArray.push_back(&chain);
    }
    return hops::computeExpectedSquaredJumpDistanceEveryKth<StateType, MatrixType>(chainPtrArray, k, sqrtCovariance, lag, start, stop, step);
}



template<typename StateType, typename MatrixType>
inline std::tuple<StateType, hops::IntermediateExpectedSquaredJumpDistanceResults<StateType, MatrixType>>
hops::computeExpectedSquaredJumpDistanceIncrementally(const std::vector<std::vector<StateType>>& chains, 
                                                      const hops::IntermediateExpectedSquaredJumpDistanceResults<StateType, MatrixType>& intermediateResult,
                                                      size_t lag, size_t start, size_t stop, size_t step) {
    std::vector<const std::vector<StateType>*> chainPtrArray;
    for (const auto& chain : chains) {
        chainPtrArray.push_back(*chain);
    }
    return hops::computeExpectedSquaredJumpDistanceIncrementally<StateType, MatrixType>(chainPtrArray, intermediateResult, lag, start, stop, step);
}

template<typename StateType, typename MatrixType>
inline std::tuple<StateType, hops::IntermediateExpectedSquaredJumpDistanceResults<StateType, MatrixType>>
hops::computeExpectedSquaredJumpDistanceIncrementally(const std::vector<std::vector<StateType>>& chains,
                                                      size_t lag, size_t start, size_t stop, size_t step) {
    std::vector<const std::vector<StateType>*> chainPtrArray;
    for (const auto& chain : chains) {
        chainPtrArray.push_back(*chain);
    }
    return hops::computeExpectedSquaredJumpDistanceIncrementally<StateType, MatrixType>(chainPtrArray, lag, start, stop, step);
}

template<typename StateType, typename MatrixType>
inline StateType hops::computeExpectedSquaredJumpDistance(const std::vector<std::vector<StateType>>& chains,
                                                          size_t lag, size_t start, size_t stop, size_t step) {
    std::vector<const std::vector<StateType>*> chainPtrArray;
    for (const auto& chain : chains) {
        chainPtrArray.push_back(*chain);
    }
    return hops::computeExpectedSquaredJumpDistance<StateType, MatrixType>(chainPtrArray, lag, start, stop, step);
}

template<typename StateType, typename MatrixType>
inline MatrixType hops::computeExpectedSquaredJumpDistanceEveryKth(const std::vector<std::vector<StateType>>& chains, size_t k,
                                                                   size_t lag, size_t start, size_t stop, size_t step) {
    std::vector<const std::vector<StateType>*> chainPtrArray;
    for (const auto& chain : chains) {
        chainPtrArray.push_back(*chain);
    }
    return hops::computeExpectedSquaredJumpDistanceEveryKth<StateType, MatrixType>(chainPtrArray, k, lag, start, stop, step);
}



template<typename StateType, typename MatrixType>
inline std::tuple<double, hops::IntermediateExpectedSquaredJumpDistanceResults<StateType, MatrixType>>
hops::computeExpectedSquaredJumpDistanceIncrementally(const std::vector<StateType>& chains, 
                                                      const MatrixType& sqrtCovariance,
                                                      const hops::IntermediateExpectedSquaredJumpDistanceResults<StateType, MatrixType>& intermediateResult,
                                                      size_t lag, size_t start, size_t stop, size_t step) {
    if (chains.size()) {
        auto[expectedSquaredJumpDistance, _intermediateResult] = hops::computeExpectedSquaredJumpDistanceIncrementally<StateType, MatrixType>({&chains}, sqrtCovariance, intermediateResult, lag, start, stop, step);
        return {expectedSquaredJumpDistance(0), _intermediateResult};
    } else {
        return {0, {StateType::Zero(chains.size()), 0, {MatrixType::Identity(0, 0), MatrixType::Zero(0), 0}}};
    }
}

template<typename StateType, typename MatrixType>
inline std::tuple<double, hops::IntermediateExpectedSquaredJumpDistanceResults<StateType, MatrixType>>
hops::computeExpectedSquaredJumpDistanceIncrementally(const std::vector<StateType>& chains,
                                                      const MatrixType& sqrtCovariance,
                                                      size_t lag, size_t start, size_t stop, size_t step) {
    if (chains.size()) {
        auto[expectedSquaredJumpDistance, intermediateResult] = hops::computeExpectedSquaredJumpDistanceIncrementally<StateType, MatrixType>({&chains}, sqrtCovariance, lag, start, stop, step);
        return {expectedSquaredJumpDistance(0), intermediateResult};
    } else {
        return {0, {StateType::Zero(chains.size()), 0, {MatrixType::Identity(0, 0), MatrixType::Zero(0), 0}}};
    }
}

template<typename StateType, typename MatrixType>
inline double hops::computeExpectedSquaredJumpDistance(const std::vector<StateType>& chains,
                                                       const MatrixType& sqrtCovariance,
                                                       size_t lag, size_t start, size_t stop, size_t step) {
    if (chains.size()) {
        return hops::computeExpectedSquaredJumpDistance<StateType, MatrixType>({&chains}, sqrtCovariance, lag, start, stop, step)(0);
    } else {
        return 0;
    }
}

template<typename StateType, typename MatrixType>
inline StateType hops::computeExpectedSquaredJumpDistanceEveryKth(const std::vector<StateType>& chains, size_t k,
                                                                  const MatrixType& sqrtCovariance,
                                                                  size_t lag, size_t start, size_t stop, size_t step) {
    if (chains.size()) {
        return hops::computeExpectedSquaredJumpDistanceEveryKth<StateType, MatrixType>({&chains}, k, sqrtCovariance, lag, start, stop, step).col(0);
    } else {
        return StateType::Zero(chains.size());
    }
}



template<typename StateType, typename MatrixType>
inline std::tuple<double, hops::IntermediateExpectedSquaredJumpDistanceResults<StateType, MatrixType>>
hops::computeExpectedSquaredJumpDistanceIncrementally(const std::vector<StateType>& chains, 
                                                      const hops::IntermediateExpectedSquaredJumpDistanceResults<StateType, MatrixType>& intermediateResult,
                                                      size_t lag, size_t start, size_t stop, size_t step) {
    if (chains.size()) {
        auto[expectedSquaredJumpDistance, _intermediateResult] = hops::computeExpectedSquaredJumpDistanceIncrementally<StateType, MatrixType>({&chains}, intermediateResult, lag, start, stop, step);
        return {expectedSquaredJumpDistance(0), _intermediateResult};
    } else {
        return {0, {StateType::Zero(chains.size()), 0, {MatrixType::Identity(0, 0), MatrixType::Zero(0), 0}}};
    }
}

template<typename StateType, typename MatrixType>
inline std::tuple<double, hops::IntermediateExpectedSquaredJumpDistanceResults<StateType, MatrixType>>
hops::computeExpectedSquaredJumpDistanceIncrementally(const std::vector<StateType>& chains,
                                                      size_t lag, size_t start, size_t stop, size_t step) {
    if (chains.size()) {
        auto[expectedSquaredJumpDistance, intermediateResult] = hops::computeExpectedSquaredJumpDistanceIncrementally<StateType, MatrixType>({&chains}, lag, start, stop, step);
        return {expectedSquaredJumpDistance(0), intermediateResult};
    } else {
        return {0, {StateType::Zero(chains.size()), 0, {MatrixType::Identity(0, 0), MatrixType::Zero(0), 0}}};
    }
}

template<typename StateType, typename MatrixType>
inline double hops::computeExpectedSquaredJumpDistance(const std::vector<StateType>& chains,
                                                       size_t lag, size_t start, size_t stop, size_t step) {
    if (chains.size()) {
        return hops::computeExpectedSquaredJumpDistance<StateType, MatrixType>({&chains}, lag, start, stop, step)(0);
    } else {
        return 0;
    }
}

template<typename StateType, typename MatrixType>
inline StateType hops::computeExpectedSquaredJumpDistanceEveryKth(const std::vector<StateType>& chains, size_t k,
                                                                  size_t lag, size_t start, size_t stop, size_t step) {
    if (chains.size()) {
        return hops::computeExpectedSquaredJumpDistanceEveryKth<StateType, MatrixType>({&chains}, k, lag, start, stop, step).col(0);
    } else {
        return MatrixType::Zero(chains.size(), 1);
    }
}


#endif //HOPS_EXPECTEDSQUAREDJUMPDISTANCE_HPP

