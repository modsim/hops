#ifndef HOPS_COVARIANCE_HPP
#define HOPS_COVARIANCE_HPP

#include <vector>

namespace hops {
    template<typename StateType, typename MatrixType>
    MatrixType computeCovariance(const std::vector<const std::vector<StateType>*>& draws, 
                                 const MatrixType& covarianceSeen, 
                                 StateType& meanSeen,
                                 unsigned long numberOfSeenDraws) {
        unsigned long numberOfChains = draws.size();
        unsigned long numberOfDraws = draws[0]->size();
        unsigned long dimension = draws[0]->at(0).size();
        unsigned long numberOfUnseenDraws = numberOfDraws - numberOfSeenDraws;
        double eta = 1.0 * numberOfSeenDraws / numberOfDraws;

        StateType meanUnseen = StateType::Zero(dimension);
        for (unsigned long i = 0; i < numberOfChains; ++i) {
            for (unsigned long j = numberOfSeenDraws; j < numberOfDraws; ++j) {
                meanUnseen += draws[i]->at(j).transpose();
                if (meanSeen.size()) {
                    meanUnseen -= meanSeen;
                }
            }
        }
        meanUnseen.array() /= numberOfDraws * numberOfChains;

        StateType mean = meanUnseen;
        if (meanSeen.size()) {
            mean += meanSeen;
        }

        MatrixType covarianceUnseen = MatrixType::Zero(dimension, dimension);
        for (unsigned long i = 0; i < numberOfChains; ++i) {
            for (unsigned long j = numberOfSeenDraws; j < numberOfDraws; ++j) {
                covarianceUnseen += (draws[i]->at(j).transpose() - mean) * (draws[i]->at(j) - mean.transpose());
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
    MatrixType computeCovariance(const std::vector<StateType>* draws, 
                                 const MatrixType& covarianceSeen, 
                                 StateType& meanSeen,
                                 unsigned long numberOfSeenDraws) {
        return computeCovariance<StateType, MatrixType>(std::vector<const std::vector<StateType>*>{draws}, 
                                                        covarianceSeen, meanSeen, numberOfSeenDraws);
    }

    template<typename StateType, typename MatrixType>
    MatrixType computeCovariance(const std::vector<std::vector<StateType>>& draws, 
                                 const MatrixType& covarianceSeen, 
                                 StateType& meanSeen,
                                 unsigned long numberOfSeenDraws) {
        std::vector<const std::vector<StateType>*> drawPtrs;
        for (auto& e : draws) {
            drawPtrs.push_back(&e);
        }
        return computeCovariance<StateType, MatrixType>(drawPtrs, covarianceSeen, meanSeen, numberOfSeenDraws);
    }

    template<typename StateType, typename MatrixType>
    MatrixType computeCovariance(const std::vector<StateType>& draws, 
                                 const MatrixType& covarianceSeen, 
                                 StateType& meanSeen,
                                 unsigned long numberOfSeenDraws) {
        return computeCovariance<StateType, MatrixType>(std::vector<const std::vector<StateType>*>{&draws}, 
                                                        covarianceSeen, meanSeen, numberOfSeenDraws);
    }

    template<typename StateType, typename MatrixType>
    MatrixType computeCovariance(const std::vector<const std::vector<StateType>*>& draws) {
        MatrixType covariance = MatrixType::Zero(0, 0);
        StateType mean = StateType::Zero(0);
        return computeCovariance<StateType, MatrixType>(draws, covariance, mean, 0);
    }

    template<typename StateType, typename MatrixType>
    MatrixType computeCovariance(const std::vector<StateType>* draws) {
        return computeCovariance<StateType, MatrixType>(std::vector<const std::vector<StateType>*>{draws});
    }

    template<typename StateType, typename MatrixType>
    MatrixType computeCovariance(const std::vector<std::vector<StateType>>& draws) {
        MatrixType covariance = MatrixType::Zero(0, 0);
        StateType mean = StateType::Zero(0);
        std::vector<const std::vector<StateType>*> drawPtrs;
        for (auto& e : draws) {
            drawPtrs.push_back(&e);
        }
        return computeCovariance<StateType, MatrixType>(drawPtrs, mean, covariance, 0);
    }

    template<typename StateType, typename MatrixType>
    MatrixType computeCovariance(const std::vector<StateType>& draws) {
        return computeCovariance<StateType, MatrixType>(std::vector<const std::vector<StateType>*>{&draws});
    }
}

#endif // HOPS_COVARIANCE_HPP
