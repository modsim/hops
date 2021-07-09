#ifndef HOPS_METROPOLISHASTINGSFILTER_HPP
#define HOPS_METROPOLISHASTINGSFILTER_HPP

#include <random>

#include <hops/FileWriter/FileWriter.hpp>
#include <hops/MarkovChain/Recorder/IsAddMessageAvailabe.hpp>
#include <hops/MarkovChain/Recorder/IsClearRecordsAvailable.hpp>
#include <hops/RandomNumberGenerator/RandomNumberGenerator.hpp>


namespace hops {
    template<typename MarkovChainProposer>
    class MetropolisHastingsFilter : public MarkovChainProposer {
    public:
        explicit MetropolisHastingsFilter(const MarkovChainProposer &markovChainImpl) : MarkovChainProposer(
                markovChainImpl) {}

        void draw(RandomNumberGenerator &randomNumberGenerator);

        double getAcceptanceRate();

        void clearRecords() {
            numberOfProposals = 0;
            numberOfAcceptedProposals = 0;
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
        double acceptanceProbability = MarkovChainProposer::computeLogAcceptanceProbability();
        if constexpr(IsAddMessageAvailable<MarkovChainProposer>::value) {
            MarkovChainProposer::addMessage("interior(");
            MarkovChainProposer::addMessage(std::isfinite(acceptanceProbability) ? "true" : "false");
            MarkovChainProposer::addMessage(")");
            MarkovChainProposer::addMessage(" alpha(");
            MarkovChainProposer::addMessage(std::to_string(std::exp(acceptanceProbability)));
            MarkovChainProposer::addMessage(") action(");
        }
        if (acceptanceChance < acceptanceProbability) {
            MarkovChainProposer::acceptProposal();
            numberOfAcceptedProposals++;
            if constexpr(IsAddMessageAvailable<MarkovChainProposer>::value) {
                MarkovChainProposer::addMessage("accept) | ");
            }
        }
        else {
            if constexpr(IsAddMessageAvailable<MarkovChainProposer>::value) {
                MarkovChainProposer::addMessage("reject) | ");
            }
        }
    }

    template<typename MarkovChainProposer>
    double MetropolisHastingsFilter<MarkovChainProposer>::getAcceptanceRate() {
        return numberOfProposals != 0 ? static_cast<double>(numberOfAcceptedProposals) / numberOfProposals :
               0;
    }
}

#endif //HOPS_METROPOLISHASTINGSFILTER_HPP
