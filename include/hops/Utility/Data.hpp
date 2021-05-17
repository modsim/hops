#ifndef HOPS_DATA_HPP
#define HOPS_DATA_HPP

#include "../FileWriter/FileWriter.hpp"
#include "../FileWriter/FileWriterFactory.hpp"
#include "../FileWriter/FileWriterType.hpp"
#include "../Diagnostics/ExpectedSquaredJumpDistance.hpp"
#include "../Diagnostics/EffectiveSampleSize.hpp"
#include "../Diagnostics/PotentialScaleReductionFactor.hpp"
#include "../MarkovChain/MarkovChain.hpp"
#include "ChainData.hpp"

#include <Eigen/Core>

#include <vector>
#include <memory>

namespace hops {
    class Data {
    public:
        Data(long dimension = 0) : dimension(dimension) {
            //
        }

        Data(const std::vector<std::shared_ptr<MarkovChain>> &markovChains, long dimension = 0) : dimension(dimension) {
            linkWithChains(markovChains);
        }

        void setDimension(long dimension) {
            this->dimension = dimension;
        }

        void linkWithChains(const std::vector<std::shared_ptr<MarkovChain>> &markovChains) {
            chains.resize(markovChains.size());
            for (size_t i = 0; i < markovChains.size(); ++i) {
                markovChains[i]->installDataObject(chains[i]);
            }
        }


        std::vector<const std::vector<double> *> getAcceptanceRates() {
            std::vector<const std::vector<double> *> acceptanceRates(chains.size());
            for (size_t i = 0; i < acceptanceRates.size(); ++i) {
                acceptanceRates[i] = chains[i].acceptanceRates.get();
            }
            return acceptanceRates;
        }

        std::vector<const std::vector<double> *> getNegativeLogLikelihood() {
            std::vector<const std::vector<double> *> negativeLogLikelihood(chains.size());
            for (size_t i = 0; i < negativeLogLikelihood.size(); ++i) {
                negativeLogLikelihood[i] = chains[i].negativeLogLikelihood.get();
            }
            return negativeLogLikelihood;
        }

        std::vector<const std::vector<Eigen::VectorXd> *> getStates() {
            std::vector<const std::vector<Eigen::VectorXd> *> states(chains.size());
            for (size_t i = 0; i < states.size(); ++i) {
                states[i] = chains[i].states.get();
            }
            return states;
        }

        std::vector<const std::vector<long> *> getTimestamps() {
            std::vector<const std::vector<long> *> timestamps(chains.size());
            for (size_t i = 0; i < timestamps.size(); ++i) {
                timestamps[i] = chains[i].timestamps.get();
            }
            return timestamps;
        }


        void computeAcceptanceRate() {
            acceptanceRate = Eigen::VectorXd(chains.size());
            for (size_t i = 0; i < chains.size(); ++i) {
                acceptanceRate(i) = chains[i].getAcceptanceRates().back();
            }
        }

        void computeExpectedSquaredJumpDistance() {
            std::vector<const std::vector<Eigen::VectorXd> *> states(chains.size());
            if (!chains.size()) {
                throw EmptyChainDataException();
            }

            for (size_t i = 0; i < states.size(); ++i) {
                states[i] = chains[i].states.get();
                if (!states[i]) {
                    throw EmptyChainDataException();
                }
            }
            std::vector<double> expectedSquaredJumpDistance = ::hops::computeExpectedSquaredJumpDistance(states);
            this->expectedSquaredJumpDistance = Eigen::Map<Eigen::VectorXd>(expectedSquaredJumpDistance.data(),
                                                                            chains.size());
        }

        void computeEffectiveSampleSize() {
            std::vector<const std::vector<Eigen::VectorXd> *> states(chains.size());
            if (!chains.size()) {
                throw EmptyChainDataException();
            }

            for (size_t i = 0; i < states.size(); ++i) {
                states[i] = chains[i].states.get();
                if (!states[i]) {
                    throw EmptyChainDataException();
                }
            }
            std::vector<double> effectiveSampleSize = ::hops::computeEffectiveSampleSize(states);
            this->effectiveSampleSize = Eigen::Map<Eigen::VectorXd>(effectiveSampleSize.data(), dimension);
        }

        void computePotentialScaleReductionFactor() {
            std::vector<const std::vector<Eigen::VectorXd> *> states(chains.size());
            if (!chains.size()) {
                throw EmptyChainDataException();
            }

            for (size_t i = 0; i < states.size(); ++i) {
                states[i] = chains[i].states.get();
                if (!states[i]) {
                    throw EmptyChainDataException();
                }
            }
            std::vector<double> potentialScaleReductionFactor = ::hops::computePotentialScaleReductionFactor(states);
            this->potentialScaleReductionFactor = Eigen::Map<Eigen::VectorXd>(potentialScaleReductionFactor.data(),
                                                                              dimension);
        }

        void computeTotalTimeTaken() {
            totalTimeTaken = Eigen::VectorXd(chains.size());
            for (size_t i = 0; i < chains.size(); ++i) {
                totalTimeTaken(i) = chains[i].getTimestamps().back() - chains[i].getTimestamps().front();
            }
        }

        void reset() {
            for (size_t i = 0; i < chains.size(); ++i) {
                chains[i].reset();
            }
        }

        void write(const std::string &outputDirectory, bool discardRawData = false,
                   FileWriterType fileWriterType = FileWriterType::CSV) const {
            if (!discardRawData) {
                for (size_t i = 0; i < chains.size(); ++i) {
                    auto fileWriter = FileWriterFactory::createFileWriter(
                            outputDirectory + "/chain" + std::to_string(i), fileWriterType);
                    chains[i].write(fileWriter.get());
                }
            }

            auto statisticsWriter = FileWriterFactory::createFileWriter(outputDirectory + "/statistics",
                                                                        fileWriterType);

            if (acceptanceRate.size() > 0) {
                statisticsWriter->write("acceptanceRate", Eigen::MatrixXd(acceptanceRate.transpose()));
            }

            if (expectedSquaredJumpDistance.size() > 0) {
                statisticsWriter->write("expectedSquaredJumpDistance",
                                        Eigen::MatrixXd(expectedSquaredJumpDistance.transpose()));
            }

            if (effectiveSampleSize.size() > 0) {
                statisticsWriter->write("effectiveSampleSize", Eigen::MatrixXd(effectiveSampleSize.transpose()));
            }

            if (potentialScaleReductionFactor.size() > 0) {
                statisticsWriter->write("potentialScaleReductionFactor",
                                        Eigen::MatrixXd(potentialScaleReductionFactor.transpose()));
            }

            if (totalTimeTaken.size() > 0) {
                statisticsWriter->write("totalTimeTaken", Eigen::MatrixXd(totalTimeTaken.transpose()));
            }

            auto tuningWriter = FileWriterFactory::createFileWriter(outputDirectory + "/tuning", fileWriterType);

            if (totalNumberOfTuningSamples > 0) {
                tuningWriter->write("totalNumberOfTuningSamples",
                                    std::vector<long>{static_cast<long>(totalNumberOfTuningSamples)});
                tuningWriter->write("tunedStepSize", std::vector<double>{tunedStepSize});
                tuningWriter->write("maximumExpectedSquaredJumpDistance",
                                    std::vector<double>{maximumExpectedSquaredJumpDistance});
            }
        }

        void setTuningData(unsigned long totalNumberOfTuningSamples, double tunedStepSize,
                           double maximumExpectedSquaredJumpDistance) {
            this->totalNumberOfTuningSamples = totalNumberOfTuningSamples;
            this->tunedStepSize = tunedStepSize;
            this->maximumExpectedSquaredJumpDistance = maximumExpectedSquaredJumpDistance;
        }

    private:
        std::vector<ChainData> chains;

        Eigen::VectorXd acceptanceRate;
        Eigen::VectorXd expectedSquaredJumpDistance;
        Eigen::VectorXd effectiveSampleSize;
        Eigen::VectorXd potentialScaleReductionFactor;
        Eigen::VectorXd totalTimeTaken;

        unsigned long totalNumberOfTuningSamples = 0;
        double tunedStepSize;
        double maximumExpectedSquaredJumpDistance;

        std::vector<std::vector<double>> sampleVariances;
        std::vector<std::vector<double>> intraChainExpectations;
        std::vector<double> interChainExpectation;
        unsigned long numSeen = 0;

        long dimension = 0;

        friend Eigen::VectorXd computeAcceptanceRate(Data &data);

        friend Eigen::VectorXd computeExpectedSquaredJumpDistance(Data &data);

        friend Eigen::VectorXd computeEffectiveSampleSize(Data &data);

        friend Eigen::VectorXd computePotentialScaleReductionFactor(Data &data);

        friend Eigen::VectorXd computeTotalTimeTaken(Data &data);
    };

    inline Eigen::VectorXd computeAcceptanceRate(Data &data) {
        data.computeAcceptanceRate();
        return data.acceptanceRate;
    }

    inline Eigen::VectorXd computeExpectedSquaredJumpDistance(Data &data) {
        data.computeExpectedSquaredJumpDistance();
        return data.expectedSquaredJumpDistance;
    }

    inline Eigen::VectorXd computeEffectiveSampleSize(Data &data) {
        data.computeEffectiveSampleSize();
        return data.effectiveSampleSize;
    }

    inline Eigen::VectorXd computePotentialScaleReductionFactor(Data &data) {
        data.computePotentialScaleReductionFactor();
        return data.potentialScaleReductionFactor;
    }

    inline Eigen::VectorXd computeTotalTimeTaken(Data &data) {
        data.computeTotalTimeTaken();
        return data.totalTimeTaken;
    }
}

#endif // HOPS_DATA_HPP

