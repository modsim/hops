#ifndef HOPS_NOOPDRAW_HPP
#define HOPS_NOOPDRAW_HPP

#include <hops/MarkovChain/Draw/IsAcceptProposalAvailable.hpp>
#include <hops/RandomNumberGenerator/RandomNumberGenerator.hpp>

namespace hops {
    template<typename MarkovChainProposer>
    class NoOpDraw : public MarkovChainProposer {
    public:
        explicit NoOpDraw(const MarkovChainProposer &markovChainImpl) : MarkovChainProposer(markovChainImpl) {}

        void draw(RandomNumberGenerator &randomNumberGenerator) {
            MarkovChainProposer::propose(randomNumberGenerator);
            if constexpr(IsAcceptProposalAvailable<MarkovChainProposer>::value) {
                MarkovChainProposer::acceptProposal();
            }
        }
    };
}

#endif //HOPS_NOOPDRAW_HPP
