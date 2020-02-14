#ifndef HOPS_MARKOVCHAINADAPTER_HPP
#define HOPS_MARKOVCHAINADAPTER_HPP

#include <hops/MarkovChain/MarkovChain.hpp>

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

        void writeHistory(const FileWriter *const fileWriter) override {
            MarkovChainImpl::writeRecordsToFile(fileWriter);
        }

        void clearHistory() override {
            MarkovChainImpl::clearRecords();
        }
    };
}

#endif //HOPS_MARKOVCHAINADAPTER_HPP
