#ifndef NUPS_NOOPDRAW_HPP
#define NUPS_NOOPDRAW_HPP

#include <nups/MarkovChain/Draw/IsAcceptProposalAvailable.hpp>
#include <nups/RandomNumberGenerator/RandomNumberGenerator.hpp>

namespace nups {
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

#endif //NUPS_NOOPDRAW_HPP
