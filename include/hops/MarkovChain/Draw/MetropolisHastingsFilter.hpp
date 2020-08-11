#ifndef HOPS_METROPOLISHASTINGSFILTER_HPP
#define HOPS_METROPOLISHASTINGSFILTER_HPP

#include <hops/FileWriter/FileWriter.hpp>
#include <hops/MarkovChain/Recorder/IsClearRecordsAvailable.hpp>
#include <hops/MarkovChain/Recorder/IsAppendToLatestMetropolisHastingsInfoRecordAvailable.hpp>
#include <hops/RandomNumberGenerator/RandomNumberGenerator.hpp>
#include <random>

namespace hops {
    template<typename MarkovChainProposer>
    class MetropolisHastingsFilter : public MarkovChainProposer {
    public:
        explicit MetropolisHastingsFilter(const MarkovChainProposer &markovChainImpl) : MarkovChainProposer(
                markovChainImpl) {}

        void draw(RandomNumberGenerator &randomNumberGenerator);

        double getAcceptanceRate();

        void resetAcceptanceRate();

        void clearRecords() {
            if constexpr(IsClearRecordsAvailable<MarkovChainProposer>::value) {
                MarkovChainProposer::clearRecords();
            }
        }

    private:
        long numberOfAcceptedProposals = 0;
        long numberOfProposals = 0;
        std::uniform_real_distribution<double> uniformRealDistribution;
    };

    template<typename MarkovChainProposer>
    void MetropolisHastingsFilter<MarkovChainProposer>::draw(hops::RandomNumberGenerator &randomNumberGenerator) {
        MarkovChainProposer::propose(randomNumberGenerator);
        numberOfProposals++;
        double acceptanceChance = std::log(uniformRealDistribution(randomNumberGenerator));
        double acceptanceProbability = MarkovChainProposer::calculateLogAcceptanceProbability();
        // TODO add logging acceptanceProbability
        if (acceptanceChance < acceptanceProbability) {
            MarkovChainProposer::acceptProposal();
            numberOfAcceptedProposals++;
            if constexpr(IsAppendToLatestMetropolisHastingsInfoRecordAvailable<MarkovChainProposer>::value) {
                MarkovChainProposer::appendToLatestMetropolisHastingsInfoRecord("(acceptance)");
            }
        } else {
            if constexpr(IsAppendToLatestMetropolisHastingsInfoRecordAvailable<MarkovChainProposer>::value) {
                MarkovChainProposer::appendToLatestMetropolisHastingsInfoRecord("(rejection)");
            }
        }
    }

    template<typename MarkovChainProposer>
    void MetropolisHastingsFilter<MarkovChainProposer>::resetAcceptanceRate() {
        numberOfAcceptedProposals = 0;
        numberOfProposals = 0;
    }

    template<typename MarkovChainProposer>
    double MetropolisHastingsFilter<MarkovChainProposer>::getAcceptanceRate() {
        return static_cast<double>(numberOfAcceptedProposals) / numberOfProposals;
    }
}

#endif //HOPS_METROPOLISHASTINGSFILTER_HPP
