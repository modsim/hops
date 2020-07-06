#ifndef HOPS_MARKOVCHAIN_HPP
#define HOPS_MARKOVCHAIN_HPP

#include <hops/RandomNumberGenerator/RandomNumberGenerator.hpp>
#include "MarkovChainAttribute.hpp"

namespace hops {
    class FileWriter;

    class MarkovChain {
    public:
        /**
         * @brief Updates internal state of the chain and stores numberOfSamples samples in memory.
         * @param randomNumberGenerator
         * @param numberOfSamples Number of samples to draw.
         */
        virtual void draw(RandomNumberGenerator &randomNumberGenerator, long numberOfSamples) = 0;

        /**
         * @brief Updates internal state of the chain and stores numberOfSamples samples in memory.
         * @param randomNumberGenerator
         * @param numberOfSamples Number of samples to draw.
         * @param thinning Number of internal state updates between every state that is stored.
         */
        virtual void draw(RandomNumberGenerator &randomNumberGenerator, long numberOfSamples, long thinning) = 0;

        /**
         * @brief Writes all stored chain history using the fileWriter.
         * @param fileWriter
         */
        virtual void writeHistory(const FileWriter *fileWriter) = 0;

        /**
         * @brief Deletes all stored chain history.
         */
        virtual void clearHistory() = 0;

        virtual std::string getName() = 0;

        virtual void setAttribute(MarkovChainAttribute markovChainAttribute, double value) = 0;

        virtual double getAttribute(MarkovChainAttribute markovChainAttribute) = 0;

        virtual double getAcceptanceRate() = 0;

        virtual ~MarkovChain() {}
    };
}

#endif //HOPS_MARKOVCHAIN_HPP
