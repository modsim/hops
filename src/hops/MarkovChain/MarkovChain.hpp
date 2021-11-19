#ifndef HOPS_MARKOVCHAIN_HPP
#define HOPS_MARKOVCHAIN_HPP

#include <Eigen/Core>
#include <memory>
#include <vector>

#include "MarkovChainAttribute.hpp"
#include <hops/RandomNumberGenerator/RandomNumberGenerator.hpp>
#include <hops/Utility/ChainData.hpp>
#include <hops/Utility/VectorType.hpp>


namespace hops {
    class FileWriter;

//
//    /**
//     * @brief Prototype for new MarkovChainInterface
//     */
//    class MarkovChainPrototype {
//    public:
//        virtual ~MarkovChainPrototype() = default;
////
////        /**
////         * @brief Updates internal state of the chain and returns a single new state as well as the acceptance rate throughout the thinning.
////         * @param randomNumberGenerator
////         * @param thinning Number of samples to draw but discard before reporting a single new sample.
////         */
////        virtual std::pair<double, MatrixType> draw(RandomNumberGenerator &randomNumberGenerator, long thinning) = 0;
////
////        virtual void setState(const MatrixType & state) = 0;
////
////        virtual MatrixType getState() = 0;
//    };


    template<typename StateType = Eigen::VectorXd>
    class MarkovChainInterface {
    public:
        virtual ~MarkovChainInterface() = default;

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
        virtual void writeHistory(FileWriter *fileWriter) = 0;

        virtual void installDataObject(ChainData &chainData) = 0;

        virtual const std::vector<StateType> &getStateRecords() = 0;

        virtual void reserveStateRecords(long numberOfSamples) = 0;

        /**
         * @brief Deletes all stored chain history.
         */
        virtual void clearHistory() = 0;

        virtual std::string getName() = 0;

        virtual void setAttribute(MarkovChainAttribute markovChainAttribute, double value) = 0;

        virtual void setState(const VectorType &state) = 0;

        virtual double getAttribute(MarkovChainAttribute markovChainAttribute) = 0;

        virtual double getAcceptanceRate() = 0;
    };

    typedef MarkovChainInterface<Eigen::VectorXd> MarkovChain;
}

#endif //HOPS_MARKOVCHAIN_HPP
