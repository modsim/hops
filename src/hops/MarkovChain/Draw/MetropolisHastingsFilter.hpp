#ifndef HOPS_METROPOLISHASTINGSFILTER_HPP
#define HOPS_METROPOLISHASTINGSFILTER_HPP

#include <random>

#include "hops/RandomNumberGenerator/RandomNumberGenerator.hpp"


namespace hops {
    template<typename MarkovChainProposer>
    class MetropolisHastingsFilter : public MarkovChainProposer {
    public:
        explicit MetropolisHastingsFilter(const MarkovChainProposer &markovChainImpl) : MarkovChainProposer(
                markovChainImpl) {}

        double draw(RandomNumberGenerator &randomNumberGenerator);

    private:
        std::uniform_real_distribution<double> uniformRealDistribution;
    };

    template<typename MarkovChainProposer>
    double MetropolisHastingsFilter<MarkovChainProposer>::draw(hops::RandomNumberGenerator &randomNumberGenerator) {
        MarkovChainProposer::propose(randomNumberGenerator);
        double acceptanceProbability = MarkovChainProposer::computeLogAcceptanceProbability();
        double acceptanceChance = std::log(uniformRealDistribution(randomNumberGenerator));

        double acceptance = 0;

        if (acceptanceChance < acceptanceProbability) {
            MarkovChainProposer::acceptProposal();
            acceptance = 1;
        }

        return acceptance;
    }
}

#endif //HOPS_METROPOLISHASTINGSFILTER_HPP
