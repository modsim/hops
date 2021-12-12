#ifndef HOPS_NOOPDRAWADAPTER_HPP
#define HOPS_NOOPDRAWADAPTER_HPP

#include "IsAcceptProposalAvailable.hpp"
#include "../../RandomNumberGenerator/RandomNumberGenerator.hpp"

namespace hops {
    template<typename MarkovChainProposer>
    class NoOpDrawAdapter : public MarkovChainProposer {
    public:
        explicit NoOpDrawAdapter(const MarkovChainProposer &markovChainImpl) : MarkovChainProposer(markovChainImpl) {}

        double draw(RandomNumberGenerator &randomNumberGenerator) {
            MarkovChainProposer::propose(randomNumberGenerator);
            MarkovChainProposer::acceptProposal();
            return 1;
        }
    };
}

#endif //HOPS_NOOPDRAWADAPTER_HPP
