#ifndef HOPS_CHAINDATA_HPP
#define HOPS_CHAINDATA_HPP

#include <hops/FileWriter/FileWriter.hpp>
#include <hops/RandomNumberGenerator/RandomNumberGenerator.hpp>
#include "Exceptions.hpp"
#include <Eigen/Core>

#include <random>
#include <vector>
#include <memory>

namespace hops {
    struct ChainData {
        ChainData (std::vector<double> acceptanceRates = std::vector<double>(),
                   std::vector<double> negativeLogLikelihood = std::vector<double>(),
                   std::vector<Eigen::VectorXd> states = std::vector<Eigen::VectorXd>(),
                   std::vector<long> timestamps = std::vector<long>()
            ) : acceptanceRates(std::make_shared<std::vector<double>>(acceptanceRates)),
                negativeLogLikelihood(std::make_shared<std::vector<double>>(negativeLogLikelihood)),
                states(std::make_shared<std::vector<Eigen::VectorXd>>(states)),
                timestamps(std::make_shared<std::vector<long>>(timestamps)) {
            //
        }


        ChainData(const ChainData& other) {
            acceptanceRates = std::make_shared<std::vector<double>>(*other.acceptanceRates);
            negativeLogLikelihood = std::make_shared<std::vector<double>>(*other.negativeLogLikelihood);
            states = std::make_shared<std::vector<Eigen::VectorXd>>(*other.states);
            timestamps = std::make_shared<std::vector<long>>(*other.timestamps);
        }


        const std::vector<double>& getAcceptanceRates() const {
            if (!acceptanceRates) throw UninitializedDataFieldException("acceptanceRates");
            return *acceptanceRates;
        }

        void setAcceptanceRates(const std::shared_ptr<std::vector<double>>& acceptanceRates) {
            this->acceptanceRates = std::shared_ptr<std::vector<double>>(acceptanceRates);
        }


        const std::vector<double>& getNegativeLogLikelihood() const {
            if (!negativeLogLikelihood) throw UninitializedDataFieldException("negativeLogLikelihood");
            return *negativeLogLikelihood;
        }

        void setNegativeLogLikelihood(const std::shared_ptr<std::vector<double>>& negativeLogLikelihood) {
            this->negativeLogLikelihood = std::shared_ptr<std::vector<double>>(negativeLogLikelihood);
        }


        const std::vector<Eigen::VectorXd>& getStates() const {
            if (!states) throw UninitializedDataFieldException("states");
            return *states;
        }

        void setStates(const std::shared_ptr<std::vector<Eigen::VectorXd>>& states) {
            this->states = std::shared_ptr<std::vector<Eigen::VectorXd>>(states);
        }


        const std::vector<long>& getTimestamps() const {
            if (!timestamps) throw UninitializedDataFieldException("timestamps");
            return *timestamps;
        }

        void setTimestamps(const std::shared_ptr<std::vector<long>> timestamps) {
            this->timestamps = std::shared_ptr<std::vector<long>>(timestamps);
        }


        void write(FileWriter *const fileWriter) const {
            if (acceptanceRates) {
                fileWriter->write("acceptanceRates", *acceptanceRates);
            }

            if (negativeLogLikelihood) {
                fileWriter->write("negativeLogLikelihood", *negativeLogLikelihood);
            }

            if (states) {
                fileWriter->write("states", *states);
            }

            if (timestamps) {
                fileWriter->write("timestamps", *timestamps);
            }
        }

        void reset() {
			if (acceptanceRates) {
                (*acceptanceRates.get()).clear();
            }

            if (negativeLogLikelihood) {
                negativeLogLikelihood->clear();
            }

            if (states) {
                states->clear();
            }

            if (timestamps) {
                timestamps->clear();
            }
		}

        ChainData thin(size_t thinning) const {
            ChainData newChainData{};

            const std::vector<double>& oldAcceptanceRatesRef = *acceptanceRates;
            const std::vector<double>& oldNegativeLogLikelihoodRef = *negativeLogLikelihood;
            const std::vector<Eigen::VectorXd>& oldStatesRef = *states;
            const std::vector<long>& oldTimestampsRef = *timestamps;

            std::vector<double>& newAcceptanceRatesRef = *newChainData.acceptanceRates;
            std::vector<double>& newNegativeLogLikelihoodRef = *newChainData.negativeLogLikelihood;
            std::vector<Eigen::VectorXd>& newStatesRef = *newChainData.states;
            std::vector<long>& newTimestampsRef = *newChainData.timestamps;

            size_t max = std::max({oldAcceptanceRatesRef.size(), oldNegativeLogLikelihoodRef.size(), oldStatesRef.size(), oldTimestampsRef.size()});

            for (size_t i = 0; i < max; i += thinning) {
                if (i < oldAcceptanceRatesRef.size()) {
                    newAcceptanceRatesRef.push_back(oldAcceptanceRatesRef[i]);
                }
                if (i < oldNegativeLogLikelihoodRef.size()) {
                    newNegativeLogLikelihoodRef.push_back(oldNegativeLogLikelihoodRef[i]);
                }
                if (i < oldStatesRef.size()) {
                    newStatesRef.push_back(oldStatesRef[i]);
                }
                if (i < oldTimestampsRef.size()) {
                    newTimestampsRef.push_back(oldTimestampsRef[i]);
                }
            }

            return newChainData;
        }

        ChainData subsample(size_t numberOfSubsamples, long seed = 0) const {
            ChainData newChainData{};
            RandomNumberGenerator rng(seed);

            const std::vector<double>& oldAcceptanceRatesRef = *acceptanceRates;
            const std::vector<double>& oldNegativeLogLikelihoodRef = *negativeLogLikelihood;
            const std::vector<Eigen::VectorXd>& oldStatesRef = *states;
            const std::vector<long>& oldTimestampsRef = *timestamps;

            std::vector<double>& newAcceptanceRatesRef = *newChainData.acceptanceRates;
            std::vector<double>& newNegativeLogLikelihoodRef = *newChainData.negativeLogLikelihood;
            std::vector<Eigen::VectorXd>& newStatesRef = *newChainData.states;
            std::vector<long>& newTimestampsRef = *newChainData.timestamps;

            size_t max = oldStatesRef.size();
            size_t j;

            std::uniform_int_distribution<size_t> uniform(0, max);

            for (size_t i = 0; i < numberOfSubsamples; ++i) {
                j = uniform(rng);

                if (j < oldAcceptanceRatesRef.size()) {
                    newAcceptanceRatesRef.push_back(oldAcceptanceRatesRef[j]);
                }
                if (j < oldNegativeLogLikelihoodRef.size()) {
                    newNegativeLogLikelihoodRef.push_back(oldNegativeLogLikelihoodRef[j]);
                }
                if (j < oldStatesRef.size()) {
                    newStatesRef.push_back(oldStatesRef[j]);
                }
                if (j < oldTimestampsRef.size()) {
                    newTimestampsRef.push_back(oldTimestampsRef[j]);
                }
            }

            return newChainData;
        }

        void append(const ChainData& other) {
            std::vector<double>& acceptanceRatesRef = *acceptanceRates;
            std::vector<double>& negativeLogLikelihoodRef = *negativeLogLikelihood;
            std::vector<Eigen::VectorXd>& statesRef = *states;
            std::vector<long>& timestampsRef = *timestamps;

            std::vector<double>& newAcceptanceRatesRef = *other.acceptanceRates;
            std::vector<double>& newNegativeLogLikelihoodRef = *other.negativeLogLikelihood;
            std::vector<Eigen::VectorXd>& newStatesRef = *other.states;
            std::vector<long>& newTimestampsRef = *other.timestamps;

            size_t max = std::max({newAcceptanceRatesRef.size(), newNegativeLogLikelihoodRef.size(), newStatesRef.size(), newTimestampsRef.size()});

            acceptanceRatesRef.reserve(acceptanceRatesRef.size() + newAcceptanceRatesRef.size());
            negativeLogLikelihoodRef.reserve(negativeLogLikelihoodRef.size() + newNegativeLogLikelihoodRef.size());
            statesRef.reserve(statesRef.size() + newStatesRef.size());
            timestampsRef.reserve(timestampsRef.size() + newTimestampsRef.size());

            for (size_t i = 0; i < max; ++i) {
                if (i < newAcceptanceRatesRef.size()) {
                    acceptanceRatesRef.push_back(newAcceptanceRatesRef[i]);
                }
                if (i < newNegativeLogLikelihoodRef.size()) {
                    negativeLogLikelihoodRef.push_back(newNegativeLogLikelihoodRef[i]);
                }
                if (i < newStatesRef.size()) {
                    statesRef.push_back(newStatesRef[i]);
                }
                if (i < newTimestampsRef.size()) {
                    timestampsRef.push_back(newTimestampsRef[i]);
                }
            }
        }


        std::shared_ptr<std::vector<double>> acceptanceRates;
        std::shared_ptr<std::vector<double>> negativeLogLikelihood;
        std::shared_ptr<std::vector<Eigen::VectorXd>> states;
        std::shared_ptr<std::vector<long>> timestamps;
    };
}

#endif // HOPS_CHAINDATA_HPP

