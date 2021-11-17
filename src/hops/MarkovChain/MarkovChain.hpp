#ifndef HOPS_MARKOVCHAIN_HPP
#define HOPS_MARKOVCHAIN_HPP

#include <hops/RandomNumberGenerator/RandomNumberGenerator.hpp>
#include <hops/Utility/VectorType.hpp>

namespace hops {
    class FileWriter;

    class MarkovChain {
    public:
        virtual ~MarkovChain() = default;

        /**
         * @brief Updates internal state of the chain and returns a single new state as well as the acceptance rate throughout the thinning.
         * @param randomNumberGenerator
         * @param thinning Number of samples to draw but discard before reporting a single new sample.
         */
        virtual std::pair<double, VectorType> draw(RandomNumberGenerator &randomNumberGenerator, long thinning) = 0;
    };

    //typedef MarkovChainInterface<Eigen::VectorXd> MarkovChain;

#endif //HOPS_MARKOVCHAIN_HPP
