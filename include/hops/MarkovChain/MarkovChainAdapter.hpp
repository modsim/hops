#ifndef HOPS_MARKOVCHAINADAPTER_HPP
#define HOPS_MARKOVCHAINADAPTER_HPP

#include "MarkovChain.hpp"
#include "../Utility/Data.hpp"

#include "IsGetColdnessAvailable.hpp"
#include "IsGetExchangeAttemptProbabilityAvailable.hpp"
#include "IsGetStepSizeAvailable.hpp"
#include "IsSetColdnessAvailable.hpp"
#include "IsSetExchangeAttemptProbabilityAvailable.hpp"
#include "IsSetStepSizeAvailable.hpp"


namespace hops {
    template<typename MarkovChainImpl>
    class MarkovChainAdapter : public MarkovChain, public MarkovChainImpl {
    public:
        explicit MarkovChainAdapter(const MarkovChainImpl &markovChainImpl) : MarkovChainImpl(markovChainImpl) {}

        void draw(RandomNumberGenerator &randomNumberGenerator, long numberOfSamples) override {
            assert(numberOfSamples > 0);
            for (long i = 0; i < numberOfSamples; ++i) {
                MarkovChainImpl::draw(randomNumberGenerator);
                MarkovChainImpl::storeRecord();
            }
        }

        void draw(RandomNumberGenerator &randomNumberGenerator, long numberOfSamples, long thinning) override {
            assert(numberOfSamples > 0);
            assert(thinning > 0);
            for (long i = 0; i < numberOfSamples; ++i) {
                for (long j = 0; j < thinning; ++j) {
                    MarkovChainImpl::draw(randomNumberGenerator);
                }
                MarkovChainImpl::storeRecord();
            }
        }

        void writeHistory(FileWriter *const fileWriter) override {
            MarkovChainImpl::writeRecordsToFile(fileWriter);
        }

        void installDataObject(ChainData& chainData) override {
            MarkovChainImpl::installDataObject(chainData);
        }

        const std::vector<Eigen::VectorXd>& getStateRecords() override {
            return MarkovChainImpl::getStateRecords();
        }

        void reserveStateRecords(long numberOfSamples) override {
            return MarkovChainImpl::reserveStateRecords(numberOfSamples);
        }

        void clearHistory() override {
            MarkovChainImpl::clearRecords();
        }

        std::string getName() override {
            return MarkovChainImpl::getName();
        }

        double getAcceptanceRate() override {
            return MarkovChainImpl::getAcceptanceRate();
        }

        void setAttribute(MarkovChainAttribute markovChainAttribute, double value) override {
            switch (markovChainAttribute) {
                case MarkovChainAttribute::STEP_SIZE: {
                    if constexpr(IsSetStepSizeAvailable<MarkovChainImpl>::value) {
                        MarkovChainImpl::setStepSize(value);
                        break;
                    }
                    throw std::runtime_error("STEP_SIZE attribute does not exist.");
                }
                case MarkovChainAttribute::PARALLEL_TEMPERING_COLDNESS: {
                    if constexpr(IsSetColdnessAvailable<MarkovChainImpl>::value) {
                        MarkovChainImpl::setColdness(value);
                        break;
                    }
                    throw std::runtime_error("PARALLEL_TEMPERING_COLDNESS attribute does not exist.");

                }
                case MarkovChainAttribute::PARALLEL_TEMPERING_EXCHANGE_PROBABILITY: {
                    if constexpr(IsSetExchangeAttemptProbabilityAvailable<MarkovChainImpl>::value) {
                        MarkovChainImpl::setExchangeAttemptProbability(value);
                        break;
                    }
                    throw std::runtime_error("PARALLEL_TEMPERING_EXCHANGE_PROBABILITY attribute does not exist.");
                }
                default: {
                    throw std::runtime_error("Attribute does not exist.");
                }
            }
        }

        void setState(Eigen::Matrix<double, -1, 1, 0, -1, 1> state) override {
           MarkovChainImpl::setState(state);
        }

        double getAttribute(MarkovChainAttribute markovChainAttribute) override {
            switch (markovChainAttribute) {
                case MarkovChainAttribute::STEP_SIZE: {
                    if constexpr(IsGetStepSizeAvailable<MarkovChainImpl>::value) {
                        return MarkovChainImpl::getStepSize();
                    }
                    throw std::runtime_error("STEP_SIZE attribute does not exist.");
                }
                case MarkovChainAttribute::PARALLEL_TEMPERING_COLDNESS: {
                    if constexpr(IsGetColdnessAvailable<MarkovChainImpl>::value) {
                        return MarkovChainImpl::getColdness();
                    }
                    throw std::runtime_error("PARALLEL_TEMPERING_COLDNESS attribute does not exist.");

                }
                case MarkovChainAttribute::PARALLEL_TEMPERING_EXCHANGE_PROBABILITY: {
                    if constexpr(IsGetExchangeAttemptProbabilityAvailable<MarkovChainImpl>::value) {
                        return MarkovChainImpl::getExchangeAttemptProbability();
                    }
                    throw std::runtime_error("PARALLEL_TEMPERING_EXCHANGE_PROBABILITY attribute does not exist.");
                }
                default: {
                    throw std::runtime_error("Attribute does not exist.");
                }
            }
        }
    };
}

#endif //HOPS_MARKOVCHAINADAPTER_HPP
