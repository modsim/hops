#ifndef HOPS_PARALLELTEMPERING_HPP
#define HOPS_PARALLELTEMPERING_HPP

#ifdef HOPS_MPI_SUPPORTED

#include <hops/FileWriter/FileWriter.hpp>
#include <hops/MarkovChain/Recorder/IsStoreRecordAvailable.hpp>
#include <hops/MarkovChain/Recorder/IsWriteRecordsToFileAvailable.hpp>
#include <hops/RandomNumberGenerator/RandomNumberGenerator.hpp>
#include "MpiInitializerFinalizer.hpp"
#include <random>

namespace hops {
    /**
     * @brief Mixin for adding parallel tempering to Markov chains. Requires MPI.
     * @tparam MarkovChainImpl A class that has the ColdnessAttribute and its Recorders mixed in.
     */
    template<typename MarkovChainImpl>
    class ParallelTempering : public MarkovChainImpl {
    public:
        ParallelTempering(const MarkovChainImpl &markovChainImpl, // NOLINT(cppcoreguidelines-pro-type-member-init)
                          RandomNumberGenerator synchronizedRandomNumberGenerator,
                          double exchangeAttemptProbability = 0.1) :
                MarkovChainImpl(markovChainImpl),
                synchronizedRandomNumberGenerator(synchronizedRandomNumberGenerator),
                exchangeAttemptProbability(exchangeAttemptProbability) {
            if (exchangeAttemptProbability > 1) {
                this->exchangeAttemptProbability = 1;
            } else if (exchangeAttemptProbability < 0) {
                this->exchangeAttemptProbability = 0;
            }

            MpiInitializerFinalizer::initializeAndQueueFinalizeAtExit();
            MPI_Comm_dup(MPI_COMM_WORLD, &communicator);
            MPI_Comm_size(communicator, &numberOfChains);

            int chainIndex;
            MPI_Comm_rank(communicator, &chainIndex);
            int largestChainIndex = numberOfChains == 1 ? 1 : numberOfChains - 1;
            MarkovChainImpl::setColdness(1. - static_cast<double>(chainIndex) / largestChainIndex);
        }

        void draw(RandomNumberGenerator &randomNumberGenerator) {
            MarkovChainImpl::draw(randomNumberGenerator);
            executeParallelTemperingStep();
        }

        void writeRecordsToFile(const FileWriter *const fileWriter) const {
            if constexpr(IsWriteRecordsToFileAvailable<MarkovChainImpl>::value) {
                int chainIndex;
                MPI_Comm_rank(communicator, &chainIndex);
                if (chainIndex == 0) {
                    MarkovChainImpl::writeRecordsToFile(fileWriter);
                }
            }
        }

        void storeRecord() {
            if constexpr(IsStoreRecordAvailable<MarkovChainImpl>::value) {
                int chainIndex;
                MPI_Comm_rank(communicator, &chainIndex);
                if (chainIndex == 0) {
                    MarkovChainImpl::storeRecord();
                }
            }
        }

        /**
         * @return true only if state exchange occurred.
         */
        bool executeParallelTemperingStep() {
            if (shouldProposeExchange()) {
                // TODO log acceptance?
                std::pair<int, int> chainPair = generateChainPairForExchangeProposal();
                int world_rank;
                MPI_Comm_rank(communicator, &world_rank);
                if (chainPair.first == world_rank || chainPair.second == world_rank) {
                    int otherChainRank = world_rank == chainPair.first ? chainPair.second : chainPair.first;

                    double acceptanceProbability = calculateExchangeAcceptanceProbability(otherChainRank);
                    double chance = uniformRealDistribution(synchronizedRandomNumberGenerator);
                    if (chance <= acceptanceProbability) {
                        exchangeStates(otherChainRank);
                        return true;
                    }
                    return false;

                } else {
                    // keeps all random number generators in sync
                    uniformRealDistribution(synchronizedRandomNumberGenerator);
                }
            }
            return false;
        }

        double calculateExchangeAcceptanceProbability(int otherChainRank) {
            double coldness = this->getColdness();
            double coldNegativeLogLikelihood = this->getNegativeLogLikelihoodOfCurrentState() / coldness;
            double thisChainProperties[] = {
                    coldness,
                    coldNegativeLogLikelihood
            };

            double otherChainProperties[2];
            std::memcpy(otherChainProperties, thisChainProperties, sizeof(double) * 2);

            MPI_Sendrecv_replace(otherChainProperties, 2, MPI_DOUBLE, otherChainRank, MpiInitializerFinalizer::getInternalMpiTag(),
                                 otherChainRank, MpiInitializerFinalizer::getInternalMpiTag(), communicator, MPI_STATUS_IGNORE);

            // 1*1=(-1)*(-1) => the signs come out consistently for both chains for the acceptance probability
            double diffColdness = thisChainProperties[0] - otherChainProperties[0];
            double diffNegativeLoglikelihoods = thisChainProperties[1] - otherChainProperties[1];

            double acceptanceProbability = std::exp(diffColdness * diffNegativeLoglikelihoods);
            return acceptanceProbability;
        }

        void exchangeStates(int otherChainRank) {
            typename MarkovChainImpl::StateType thisState = MarkovChainImpl::getState();

            MPI_Sendrecv_replace(thisState.data(), thisState.size(), MPI_DOUBLE, otherChainRank, MpiInitializerFinalizer::getInternalMpiTag(),
                                 otherChainRank, MpiInitializerFinalizer::getInternalMpiTag(), communicator, MPI_STATUS_IGNORE);

            MarkovChainImpl::setState(thisState);
        }

        std::pair<int, int> generateChainPairForExchangeProposal() {

            int chainIndex = uniformIntDistribution(synchronizedRandomNumberGenerator,
                                                    std::uniform_int_distribution<int>::param_type(0,
                                                                                                   numberOfChains - 2));
            return std::make_pair(chainIndex, chainIndex + 1);
        }

        bool shouldProposeExchange() {
            double chance = uniformRealDistribution(synchronizedRandomNumberGenerator);
            return (chance < exchangeAttemptProbability);
        }

        double getExchangeAttemptProbability() const {
            return exchangeAttemptProbability;
        }

        void setExchangeAttemptProbability(double newExchangeAttemptProbability) {
            ParallelTempering::exchangeAttemptProbability = newExchangeAttemptProbability;
        }

    private:
        int numberOfChains;
        double exchangeAttemptProbability;
        std::uniform_int_distribution<int> uniformIntDistribution;
        std::uniform_real_distribution<double> uniformRealDistribution;
        MPI_Comm communicator;
        RandomNumberGenerator synchronizedRandomNumberGenerator;
    };
}

#else

namespace hops {
    /**
     * @brief Mixin for adding parallel tempering to Markov chains. Requires MPI.
     * @tparam MarkovChainImpl A class that has the ColdnessAttribute and its Recorders mixed in.
     */
    template<typename MarkovChainImpl>
    class ParallelTempering : public MarkovChainImpl {
    public:
        ParallelTempering(const MarkovChainImpl &markovChainImpl // NOLINT(cppcoreguidelines-pro-type-member-init)
                          ) : MarkovChainImpl(markovChainImpl) {
            throw std::runtime_error("MPI not supported on current platform");
        }
        ParallelTempering(const MarkovChainImpl &markovChainImpl, // NOLINT(cppcoreguidelines-pro-type-member-init)
                          double) : MarkovChainImpl(markovChainImpl) {
            throw std::runtime_error("MPI not supported on current platform");
        }
    };
}
#endif //HOPS_MPI_SUPPORTED

#endif //HOPS_PARALLELTEMPERING_HPP
