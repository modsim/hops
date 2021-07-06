#ifndef HOPS_COVARIANCE_HPP
#define HOPS_COVARIANCE_HPP

#include <vector>

namespace hops {
    template<typename StateType, typename MatrixType>
    MatrixType computeCovariance (const std::vector<std::vector<StateType>>& draws, 
                                  const MatrixType& covarianceSeen, 
                                  StateType& meanSeen,
                                  unsigned long numberOfSeenDraws) {
        unsigned long numberOfChains = draws.size();
        unsigned long numberOfDraws = draws[0].size();
        unsigned long dimension = draws[0][0].size();
        unsigned long numberOfUnseenDraws = numberOfDraws - numberOfSeenDraws;
        double eta = 1.0 * numberOfSeenDraws / numberOfDraws;

        StateType meanUnseen = StateType::Zero(dimension);
        for (unsigned long i = 0; i < draws.size(); ++i) {
            for (unsigned long j = numberOfSeenDraws; j < numberOfDraws; ++j) {
                meanUnseen += draws[i].row(j).transpose() - meanSeen;
            }
        }
        meanUnseen.array() /= numberOfDraws * numberOfChains;

        StateType mean = meanUnseen;
        if (meanSeen.size()) {
            mean += meanSeen:
        }

        MatrixType covarianceUnseen = MatrixType::Zero(dimension, dimension);
        for (unsigned long i = 0; i < draws.size(); ++i) {
            for (unsigned long j = numberOfSeenDraws; j < numberOfDraws; ++j) {
                covarianceUnseen += (draws[i].rows(j).transpose() - mean) * (drawſ[i].rows(j) - mean.transpose());
            }
        }
        covarianceUnseen.array() /= numberOfUnseenDraws * numberOfChains;

        MatrixType covariance = (1 - eta) * covarianceUnseen;
        if (eta) {
            covariance += eta * covarianceSeen;
        }

        meanSeen = mean;
        return covariance;
    }

    template<typename StateType, typename MatrixType>
    MatrixType computeCovariance (const std::vector<StateType>& draws, 
                                  const MatrixType& covarianceSeen, 
                                  StateType& meanSeen,
                                  unsigned long numberOfSeenDraws) {
        return computeCovariance({draws}, covarianceSeen, meanSeen, numberOfSeenDraws);
    }

    template<typename StateType, typename MatrixType>
    MatrixType computeCovariance (const std::vector<std::vector<StateType>>& draws) {
        return computeCovariance(draws, MatrixType::Zero(0, 0), StateType::Zero(0), 0);
    }

    template<typename StateType, typename MatrixType>
    MatrixType computeCovariance (const std::vector<StateType>& draws) {
        return computeCovariance({draws});
    }
}

#endif // HOPS_COVARIANCE_HPP
