#ifndef HOPS_COVARIANCE_HPP
#define HOPS_COVARIANCE_HPP

#include <cassert>
#include <limits>
#include <tuple>
#include <vector>

namespace hops {
    // intermediate results needed to compute the covariance incrementally are:
    //      covariance matrix of the already seen states,
    //      mean vector of the already seen states 
    //      number of states already seen,
    template<typename StateType, typename MatrixType>
    using IntermediateCovarianceResults = std::tuple<MatrixType, StateType, unsigned long>;


    //MatrixType computeCovariance(const std::vector<const std::vector<StateType>*>& draws, 
    //                             const MatrixType& covarianceSeen, 
    //                             StateType& meanSeen,
    //                             unsigned long numberOfSeenDraws) {
    template<typename StateType, typename MatrixType>
    std::tuple<MatrixType, IntermediateCovarianceResults<StateType, MatrixType>> 
    computeCovarianceIncrementally(const std::vector<const std::vector<StateType>*>& draws, 
                                   const IntermediateCovarianceResults<StateType, MatrixType>& intermediateResult,
                                   size_t start = 0, size_t stop = std::numeric_limits<size_t>::max(), size_t step = 1) {
        const auto& [covarianceSeen, meanSeen, numberOfSeenDraws] = intermediateResult;

        unsigned long numberOfChains = draws.size();
        unsigned long numberOfDraws = draws[0]->size();
        unsigned long dimension = draws[0]->at(0).size();

        unsigned long _start = std::max(numberOfSeenDraws, static_cast<unsigned long>(start));
        unsigned long _stop = std::min(numberOfDraws, static_cast<unsigned long>(stop));

        unsigned long numberOfUnseenDraws = _stop - _start;
        double eta = 1.0 * numberOfSeenDraws / (numberOfUnseenDraws + numberOfSeenDraws);

        assert((covarianceSeen.size() > 0) == (eta > 0) && (meanSeen.size() > 0) == (eta > 0) && 
                "number of seen draws not zero, but previous covariance and/or mean result is empty");

        StateType meanUnseen = StateType::Zero(dimension);
        for (unsigned long i = 0; i < numberOfChains; ++i) {
            const std::vector<StateType>& _draws = *draws[i];
            for (unsigned long j = _start; j < _stop; j += step) {
                meanUnseen += _draws[j];
            }
        }
        meanUnseen.array() /= static_cast<typename StateType::Scalar>(numberOfUnseenDraws * numberOfChains);

        StateType mean = (1 - eta) * meanUnseen;
        if (eta) {
            mean += eta * meanSeen;
        }

        MatrixType covarianceUnseen = MatrixType::Zero(dimension, dimension);
        for (unsigned long i = 0; i < numberOfChains; ++i) {
            const std::vector<StateType>& _draws = *draws[i];
            for (unsigned long j = _start; j < _stop; j += step) {
                StateType centered = _draws[j] - meanUnseen;
                covarianceUnseen += centered * centered.transpose();
            }
        }
        covarianceUnseen.array() /= static_cast<typename MatrixType::Scalar>(numberOfUnseenDraws * numberOfChains);

        MatrixType covariance = (1 - eta) * covarianceUnseen;
        covariance += (1 - eta) * (meanUnseen * meanUnseen.transpose());
        if (eta) {
            covariance += eta * covarianceSeen;
            covariance += eta * (meanSeen * meanSeen.transpose());
        }
        covariance -= mean * mean.transpose();

        return {covariance, {covariance, mean, numberOfUnseenDraws + numberOfSeenDraws}};
    }

    template<typename StateType, typename MatrixType>
    std::tuple<MatrixType, IntermediateCovarianceResults<StateType, MatrixType>> 
    computeCovarianceIncrementally(const std::vector<const std::vector<StateType>*>& draws,
                                   size_t start = 0, size_t stop = std::numeric_limits<size_t>::max(), size_t step = 1) {
        IntermediateCovarianceResults<StateType, MatrixType> intermediateResult = std::make_tuple(MatrixType::Zero(0, 0), StateType::Zero(0), 0);
        return computeCovarianceIncrementally<StateType, MatrixType>(draws, intermediateResult, start, stop, step);
    }

    template<typename StateType, typename MatrixType>
    MatrixType computeCovariance(const std::vector<const std::vector<StateType>*>& draws,
                                 size_t start = 0, size_t stop = std::numeric_limits<size_t>::max(), size_t step = 1) {
        return std::get<0>(computeCovarianceIncrementally<StateType, MatrixType>(draws, start, stop, step));
    }



    template<typename StateType, typename MatrixType>
    std::tuple<MatrixType, IntermediateCovarianceResults<StateType, MatrixType>> 
    computeCovarianceIncrementally(const std::vector<std::vector<StateType>>& draws, 
                                   const IntermediateCovarianceResults<StateType, MatrixType>& intermediateResult,
                                   size_t start = 0, size_t stop = std::numeric_limits<size_t>::max(), size_t step = 1) {
        std::vector<const std::vector<StateType>*> drawPtrs;
        for (auto& e : draws) {
            drawPtrs.push_back(&e);
        }
        return computeCovarianceIncrementally<StateType, MatrixType>(drawPtrs, intermediateResult, start, stop, step);
    }

    template<typename StateType, typename MatrixType>
    std::tuple<MatrixType, IntermediateCovarianceResults<StateType, MatrixType>> 
    computeCovarianceIncrementally(const std::vector<std::vector<StateType>>& draws,
                                   size_t start = 0, size_t stop = std::numeric_limits<size_t>::max(), size_t step = 1) {
        IntermediateCovarianceResults<StateType, MatrixType> intermediateResult = std::make_tuple(MatrixType::Zero(0, 0), StateType::Zero(0), 0);
        return computeCovarianceIncrementally<StateType, MatrixType>(draws, intermediateResult, start, stop, step);
    }

    template<typename StateType, typename MatrixType>
    MatrixType computeCovariance(const std::vector<std::vector<StateType>>& draws,
                                               size_t start = 0, size_t stop = std::numeric_limits<size_t>::max(), size_t step = 1) {
        return std::get<0>(computeCovarianceIncrementally<StateType, MatrixType>(draws, start, stop, step));
    }



    template<typename StateType, typename MatrixType>
    std::tuple<MatrixType, IntermediateCovarianceResults<StateType, MatrixType>> 
    computeCovarianceIncrementally(const std::vector<StateType>& draws, 
                                   const IntermediateCovarianceResults<StateType, MatrixType>& intermediateResult,
                                   size_t start = 0, size_t stop = std::numeric_limits<size_t>::max(), size_t step = 1) {
        return computeCovarianceIncrementally<StateType, MatrixType>({&draws}, intermediateResult, start, stop, step);
    }

    template<typename StateType, typename MatrixType>
    std::tuple<MatrixType, IntermediateCovarianceResults<StateType, MatrixType>> 
    computeCovarianceIncrementally(const std::vector<StateType>& draws,
                                   size_t start = 0, size_t stop = std::numeric_limits<size_t>::max(), size_t step = 1) {
        return computeCovarianceIncrementally<StateType, MatrixType>({&draws}, start, stop, step);
    }

    template<typename StateType, typename MatrixType>
    MatrixType computeCovariance(const std::vector<StateType>& draws,
                                 size_t start = 0, size_t stop = std::numeric_limits<size_t>::max(), size_t step = 1) {
        return std::get<0>(computeCovarianceIncrementally<StateType, MatrixType>({&draws}, start, stop, step));
    }
}

#endif // HOPS_COVARIANCE_HPP
