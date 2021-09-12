#ifndef HOPS_DATA_HPP
#define HOPS_DATA_HPP

#include <hops/FileWriter/FileWriter.hpp>
#include <hops/FileWriter/FileWriterFactory.hpp>
#include <hops/FileWriter/FileWriterType.hpp>
#include <hops/MarkovChain/MarkovChain.hpp>
#include <hops/Statistics/ExpectedSquaredJumpDistance.hpp>
#include <hops/Statistics/EffectiveSampleSize.hpp>
#include <hops/Statistics/PotentialScaleReductionFactor.hpp>
#include <hops/Utility/ChainData.hpp>

#include <Eigen/Core>

#include <vector>
#include <memory>

namespace hops {
    struct Data {
        Data(long dimension = 0) : dimension(dimension) {
            //
        }

        Data(const std::vector<std::shared_ptr<MarkovChain>>& markovChains, long dimension = 0) : dimension(dimension) {
            linkWithChains(markovChains);
        }

        void setDimension(long dimension) {
            this->dimension = dimension;
        }

        void linkWithChains(const std::vector<std::shared_ptr<MarkovChain>>& markovChains) {
            chains.resize(markovChains.size());
            for (size_t i = 0; i < markovChains.size(); ++i) {
                markovChains[i]->installDataObject(chains[i]); 
            }
        }


        std::vector<const std::vector<double>*> getAcceptanceRates() {
            std::vector<const std::vector<double>*> acceptanceRates(chains.size());
            for (size_t i = 0; i < acceptanceRates.size(); ++i) {
                acceptanceRates[i] = chains[i].acceptanceRates.get();
            }
            return acceptanceRates;
        }

        std::vector<const std::vector<double>*> getNegativeLogLikelihood() {
            std::vector<const std::vector<double>*> negativeLogLikelihood(chains.size());
            for (size_t i = 0; i < negativeLogLikelihood.size(); ++i) {
                negativeLogLikelihood[i] = chains[i].negativeLogLikelihood.get();
            }
            return negativeLogLikelihood;
        }

        std::vector<const std::vector<Eigen::VectorXd>*> getStates() {
            std::vector<const std::vector<Eigen::VectorXd>*> states(chains.size());
            for (size_t i = 0; i < states.size(); ++i) {
                states[i] = chains[i].states.get();
                if (i == 0 && states[i]->size() > 0) {
                    dimension = states[i]->at(0).size();
                }
            }
            return states;
        }

        std::vector<const std::vector<long>*> getTimestamps() {
            std::vector<const std::vector<long>*> timestamps(chains.size());
            for (size_t i = 0; i < timestamps.size(); ++i) {
                timestamps[i] = chains[i].timestamps.get();
            }
            return timestamps;
        }


        void setAcceptanceRates(const std::vector<std::vector<double>>& acceptanceRates) {
            for (size_t i = 0; i < acceptanceRates.size(); ++i) {
                if (i >= chains.size()) {
                    chains.push_back(ChainData());
                }
                chains[i].acceptanceRates = std::make_shared<std::vector<double>>(acceptanceRates[i]);
            }
        }

        void setNegativeLogLikelihood(const std::vector<std::vector<double>>& negativeLogLikelihood) {
            for (size_t i = 0; i < negativeLogLikelihood.size(); ++i) {
                if (i >= chains.size()) {
                    chains.push_back(ChainData());
                }
                chains[i].negativeLogLikelihood = std::make_shared<std::vector<double>>(negativeLogLikelihood[i]);
            }
        }

        void setStates(const std::vector<std::vector<Eigen::VectorXd>>& states) {
            for (size_t i = 0; i < states.size(); ++i) {
                if (i >= chains.size()) {
                    chains.push_back(ChainData());
                }
                if (i == 0 && states[i].size() > 0) {
                    dimension = states[i].at(0).size();
                }
                chains[i].states = std::make_shared<std::vector<Eigen::VectorXd>>(states[i]);
            }
        }

        void setTimestamps(const std::vector<std::vector<long>>& timestamps) {
            for (size_t i = 0; i < timestamps.size(); ++i) {
                if (i >= chains.size()) {
                    chains.push_back(ChainData());
                }
                chains[i].timestamps = std::make_shared<std::vector<long>>(timestamps[i]);
            }
        }


        void reset() {
            for (size_t i = 0; i < chains.size(); ++i) {
                chains[i].reset();
            }
        }

        void write(const std::string& outputDirectory, bool discardRawData = false) const {
            if (!discardRawData) {
                for (size_t i = 0; i < chains.size(); ++i) {
                    auto fileWriter = FileWriterFactory::createFileWriter(outputDirectory + "/chain" + std::to_string(i), FileWriterType::CSV);
                    chains[i].write(fileWriter.get());
                }
            }

			auto statisticsWriter = FileWriterFactory::createFileWriter(outputDirectory + "/statistics", FileWriterType::CSV);

            if (acceptanceRate.size() > 0) {
                statisticsWriter->write("acceptanceRate", Eigen::MatrixXd(acceptanceRate.transpose()));
            }

            if (expectedSquaredJumpDistance.size() > 0) {
                statisticsWriter->write("expectedSquaredJumpDistance", Eigen::MatrixXd(expectedSquaredJumpDistance.transpose()));
            }

            if (effectiveSampleSize.size() > 0) {
                statisticsWriter->write("effectiveSampleSize", Eigen::MatrixXd(effectiveSampleSize.transpose()));
            }

            if (potentialScaleReductionFactor.size() > 0) {
                statisticsWriter->write("potentialScaleReductionFactor", Eigen::MatrixXd(potentialScaleReductionFactor.transpose()));
            }

            if (totalNumberOfSamples > 0) {
                statisticsWriter->write("totalNumberOfSamples", Eigen::MatrixXd(totalNumberOfSamples * Eigen::MatrixXd::Identity(1,1)));
            }

            if (totalTimeTaken.size() > 0) {
                statisticsWriter->write("totalTimeTaken", Eigen::MatrixXd(totalTimeTaken.transpose()));
            }

            if (totalNumberOfTuningSamples > 0) {
                auto tuningWriter = FileWriterFactory::createFileWriter(outputDirectory + "/tuning", FileWriterType::CSV);
                tuningWriter->write("totalNumberOfTuningSamples", std::vector<long>{static_cast<long>(totalNumberOfTuningSamples)});
                tuningWriter->write("stepSize", std::vector<double>{tunedStepSize});
                tuningWriter->write("objectiveValue", std::vector<double>{tunedObjectiveValue});
                tuningWriter->write("totalTimeTaken", std::vector<double>{totalTuningTimeTaken});

                if (tuningData.size() > 0) {
                    tuningWriter->write("data", tuningData);
                }

                if (tuningPosterior.size() > 0) {
                    tuningWriter->write("posterior", tuningPosterior);
                }
            }
        }

        void setTuningMethod(const std::string& tuningMethod) {
            this->tuningMethod = tuningMethod;
        }

        void setTotalNumberOfTuningSamples(unsigned long totalNumberOfTuningSamples) {
            this->totalNumberOfTuningSamples = totalNumberOfTuningSamples;
        }

        void setTunedStepSize(double tunedStepSize) {
            this->tunedStepSize = tunedStepSize;
        }

        void setTunedObjectiveValue(double tunedObjectiveValue) {
            this->tunedObjectiveValue = tunedObjectiveValue;
        }

        void setTotalTuningTimeTaken(double totalTuningTimeTaken) {
            this->totalTuningTimeTaken = totalTuningTimeTaken;
        }

        void setTuningData(const Eigen::MatrixXd& tuningData) {
            this->tuningData = tuningData;
        }

        void setTuningPosterior(const Eigen::MatrixXd& tuningPosterior) {
            this->tuningPosterior = tuningPosterior;
        }


        Data thin(size_t thinning) {
            Data newData{};
            for (const auto& chain : chains) {
                newData.chains.push_back(chain.thin(thinning));
            }
            return newData;
        }

        Data subsample(size_t numberOfSubsamples, size_t numberOfChains = 0) {
            Data newData{};
            numberOfChains = ( numberOfChains ? numberOfChains : chains.size() );
            for (size_t i = 0; i < numberOfChains; ++i) {
                newData.chains.push_back(chains[i].subsample(numberOfSubsamples));
            }
            return newData;
        }

        Data flatten() {
            Data newData{};
            newData.chains.push_back(chains[0]);
            for (size_t i = 1; i < chains.size(); ++i) {
                newData.chains[0].append(chains[i]);
            }
            return newData;
        }

        std::vector<ChainData> chains;

        double totalNumberOfSamples;
        Eigen::VectorXd acceptanceRate;
        Eigen::VectorXd expectedSquaredJumpDistance;
        Eigen::VectorXd effectiveSampleSize;
        Eigen::VectorXd potentialScaleReductionFactor;
        Eigen::VectorXd totalTimeTaken;

        // tuning data
        std::string tuningMethod;
        unsigned long totalNumberOfTuningSamples = 0;
        unsigned long totalNumberOfTuningIterations = 0;
        double tunedStepSize;
        double tunedObjectiveValue;
        double totalTuningTimeTaken;

        Eigen::MatrixXd tuningData;
        Eigen::MatrixXd tuningPosterior;

        std::vector<std::vector<double>> sampleVariances;
        std::vector<std::vector<double>> intraChainExpectations;
        std::vector<double> interChainExpectation;
        unsigned long numSeen = 0;

        long dimension = 0;
    };

    Eigen::VectorXd computeAcceptanceRate(Data& data);
    Eigen::VectorXd computeEffectiveSampleSize(Data& data);

    Eigen::VectorXd computeExpectedSquaredJumpDistance(const Data& data, const Eigen::MatrixXd& sqrtCovariance);
    Eigen::VectorXd computeExpectedSquaredJumpDistance(const Data& data);
    
    Eigen::VectorXd computePotentialScaleReductionFactor(Data& data);
    Eigen::VectorXd computeTotalTimeTaken(Data& data);
    long computeTotalNumberOfSamples(Data& data);

    using IntermediateAcceptanceRateResults = double;
    using IntermediateExpectedSquaredJumpDistanceResults_ = IntermediateExpectedSquaredJumpDistanceResults<Eigen::VectorXd, Eigen::MatrixXd>;
    using IntermediateTotalTimeTakenResults = Eigen::VectorXd;
    using IntermediateTotalNumberOfSamplesResults = long;

    std::tuple<Eigen::VectorXd, IntermediateAcceptanceRateResults> 
    computeAcceptanceRateIncrementally(Data& data, const IntermediateAcceptanceRateResults& intermediateResults);

    std::tuple<Eigen::VectorXd, IntermediateEffectiveSampleSizeResults> 
    computeEffectiveSampleSizeIncrementally(Data& data, const IntermediateEffectiveSampleSizeResults& intermediateResults);
    

    std::tuple<Eigen::VectorXd, IntermediateExpectedSquaredJumpDistanceResults_>
    computeExpectedSquaredJumpDistanceIncrementally(const Data& data, const Eigen::MatrixXd& sqrtCovariance);
    
    std::tuple<Eigen::VectorXd, IntermediateExpectedSquaredJumpDistanceResults_>
    computeExpectedSquaredJumpDistanceIncrementally(const Data& data);
    
    std::tuple<Eigen::VectorXd, IntermediateExpectedSquaredJumpDistanceResults_>
    computeExpectedSquaredJumpDistanceIncrementally(const Data& data, 
                                                    const IntermediateExpectedSquaredJumpDistanceResults_& intermediateResults,
                                                    const Eigen::MatrixXd& sqrtCovariance);
    
    std::tuple<Eigen::VectorXd, IntermediateExpectedSquaredJumpDistanceResults_>
    computeExpectedSquaredJumpDistanceIncrementally(const Data& data, 
                                                    const IntermediateExpectedSquaredJumpDistanceResults_& intermediateResults);
    

    std::tuple<Eigen::VectorXd, IntermediatePotentialScaleReductionFactorResults> 
    computePotentialScaleReductionFactorIncrementally(Data& data, const IntermediatePotentialScaleReductionFactorResults& intermediateResults);
    
    std::tuple<Eigen::VectorXd, IntermediateTotalTimeTakenResults> 
    computeTotalTimeTakenIncrementally(Data& data, const IntermediateTotalTimeTakenResults& intermediateResults);
    
    std::tuple<long, IntermediateTotalNumberOfSamplesResults> 
    computeTotalNumberOfSamplesIncrementally(Data& data, const IntermediateTotalNumberOfSamplesResults& intermediateResults);

    Eigen::MatrixXd computeAcceptanceRateEvery(Data& data, size_t k);
    Eigen::MatrixXd computeEffectiveSampleSizeEvery(Data& data, size_t k);

    Eigen::MatrixXd computeExpectedSquaredJumpDistanceEvery(const Data& data, size_t k, const Eigen::MatrixXd& sqrtCovariance);
    Eigen::MatrixXd computeExpectedSquaredJumpDistanceEvery(const Data& data, size_t k);

    Eigen::MatrixXd computePotentialScaleReductionFactorEvery(Data& data, size_t k);
    Eigen::MatrixXd computeTotalTimeTakenEvery(Data& data, size_t k);
}

#endif // HOPS_DATA_HPP

