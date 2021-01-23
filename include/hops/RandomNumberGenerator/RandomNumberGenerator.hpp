#ifndef HOPS_RANDOMNUMBERGENERATOR_HPP
#define HOPS_RANDOMNUMBERGENERATOR_HPP

#include <hops/thirdparty/pcg-cpp/pcg_random.hpp>

namespace hops {
    // TODO consider moving randomNumberGenerator directory into markov chain
    using RandomNumberGenerator = pcg64;
}

#endif //HOPS_RANDOMNUMBERGENERATOR_HPP
