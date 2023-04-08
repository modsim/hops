#ifndef HOPS_METROPOLISHASTINGSFILTER_HPP
#define HOPS_METROPOLISHASTINGSFILTER_HPP

#include <random>

#include "hops/FileWriter/FileWriter.hpp"
#include "hops/MarkovChain/Recorder/IsAddMessageAvailabe.hpp"
#include "hops/MarkovChain/Recorder/IsClearRecordsAvailable.hpp"
#include "hops/RandomNumberGenerator/RandomNumberGenerator.hpp"
#include "hops/Utility/VectorType.hpp"


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
