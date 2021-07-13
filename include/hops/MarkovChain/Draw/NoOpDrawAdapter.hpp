#ifndef HOPS_NOOPDRAWADAPTER_HPP
#define HOPS_NOOPDRAWADAPTER_HPP

#include "IsAcceptProposalAvailable.hpp"
#include "../../RandomNumberGenerator/RandomNumberGenerator.hpp"

namespace hops {
    template<typename MarkovChainProposer>
    class NoOpDrawAdapter : public MarkovChainProposer {
    public:
        explicit NoOpDrawAdapter(const MarkovChainProposer &markovChainImpl) : MarkovChainProposer(markovChainImpl) {}

        constexpr double getAcceptanceRate() {
            return 1.;
        }

        void draw(RandomNumberGenerator &randomNumberGenerator) {
            MarkovChainProposer::propose(randomNumberGenerator);
            if constexpr(IsAcceptProposalAvailable<MarkovChainProposer>::value) {
                MarkovChainProposer::acceptProposal();
            }
        }
    };
}

#endif //HOPS_NOOPDRAWADAPTER_HPP
