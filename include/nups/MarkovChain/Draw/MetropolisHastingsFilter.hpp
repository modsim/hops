#ifndef NUPS_METROPOLISHASTINGSFILTER_HPP
#define NUPS_METROPOLISHASTINGSFILTER_HPP

#include <nups/FileWriter/FileWriter.hpp>
#include <nups/RandomNumberGenerator/RandomNumberGenerator.hpp>
#include <random>

namespace nups {
    template<typename MarkovChainProposer>
    class MetropolisHastingsFilter : public MarkovChainProposer {
    public:
        explicit MetropolisHastingsFilter(const MarkovChainProposer &markovChainImpl) : MarkovChainProposer(
                markovChainImpl) {}

        void draw(RandomNumberGenerator &randomNumberGenerator);

        double getAcceptanceRate();

        long numberOfProposals = 0;
        long numberOfAcceptedProposals = 0;
        std::uniform_real_distribution<double> uniformRealDistribution;
    };

    template<typename MarkovChainProposer>
    void MetropolisHastingsFilter<MarkovChainProposer>::draw(nups::RandomNumberGenerator &randomNumberGenerator) {
        MarkovChainProposer::propose(randomNumberGenerator);
        numberOfProposals++;
        double acceptanceChance = std::log(uniformRealDistribution(randomNumberGenerator));
        double acceptanceProbability = MarkovChainProposer::calculateLogAcceptanceProbability();
        if (acceptanceChance < acceptanceProbability) {
            MarkovChainProposer::acceptProposal();
            numberOfAcceptedProposals++;
        }
    }

    template<typename MarkovChainProposer>
    double MetropolisHastingsFilter<MarkovChainProposer>::getAcceptanceRate() {
        return static_cast<double>(numberOfAcceptedProposals) / numberOfProposals;
    }
}

#endif //NUPS_METROPOLISHASTINGSFILTER_HPP
