#include <gtest/gtest.h>
#include <hops/MarkovChain/AcceptanceRateTuner.hpp>
#include <cmath>
#include <hops/MarkovChain/MarkovChainAdapter.hpp>
#include <hops/MarkovChain/Draw/MetropolisHastingsFilter.hpp>
#include <hops/MarkovChain/Recorder/StateRecorder.hpp>

namespace {
    class ProposerMock {
    public:
        explicit ProposerMock(double stepSize) : stepSize(stepSize) {}

        using StateType = double;

        void propose(hops::RandomNumberGenerator) { numberOfStepsTaken++; }

        void acceptProposal() {};

        [[nodiscard]] double calculateLogAcceptanceProbability() const {
            return std::log(1 - stepSize);
        };

        [[nodiscard]] StateType getState() const { return 0; };

        [[nodiscard]] std::string getName() const { return "ProposerMock"; };

        void setStepSize(double newStepSize) {
            stepSize = newStepSize;
        }

        [[nodiscard]] double getStepSize() const {
            return stepSize;
        }

        long getNumberOfStepsTaken() const {
            return numberOfStepsTaken;
        }

    private:
        double stepSize;
        long numberOfStepsTaken = 0;
    };

    TEST(AcceptanceRateTuner, nothingToTune) {
        double startingStepSize = 0.2;
        auto *markovChain
                = new hops::MarkovChainAdapter(
                        hops::StateRecorder(hops::MetropolisHastingsFilter(ProposerMock(startingStepSize))));
        hops::RandomNumberGenerator generator(42);

        double upperLimitAcceptanceRate = 0.85;
        double lowerLimitAcceptanceRate = 0.8;
        double lowerLimitStepSize = 0;
        double upperLimitStepSize = 1;
        size_t iterationsToTestStepSize = 100;
        size_t maxIterations = 1000;

        bool isTuned = hops::AcceptanceRateTuner::tune(markovChain, generator, {lowerLimitAcceptanceRate,
                                                                                upperLimitAcceptanceRate,
                                                                                lowerLimitStepSize,
                                                                                upperLimitStepSize,
                                                                                iterationsToTestStepSize,
                                                                                maxIterations});


        auto actualAcceptanceRate = markovChain->getAcceptanceRate();
        EXPECT_TRUE(isTuned);
        EXPECT_LE(markovChain->getNumberOfStepsTaken(), maxIterations + iterationsToTestStepSize);
        EXPECT_LE(actualAcceptanceRate, upperLimitAcceptanceRate);
        EXPECT_GE(actualAcceptanceRate, lowerLimitAcceptanceRate);
    }

    TEST(AcceptanceRateTuner, throwsExceptionIfUpperLimitAcceptanceRateIsLessThanLowerLimitAcceptanceRate) {
        double startingStepSize = 0.2;
        auto *markovChain
                = new hops::MarkovChainAdapter(
                        hops::StateRecorder(hops::MetropolisHastingsFilter(ProposerMock(startingStepSize))));
        hops::RandomNumberGenerator generator(42);

        double upperLimitAcceptanceRate = 0;
        double lowerLimitAcceptanceRate = 1;
        double lowerLimitStepSize = 0;
        double upperLimitStepSize = 1;
        size_t iterationsToTestStepSize = 100;
        size_t maxIterations = 1000;

        EXPECT_THROW({
                         try {
                             hops::AcceptanceRateTuner::tune(markovChain,
                                                             generator,
                                                             {lowerLimitAcceptanceRate,
                                                              upperLimitAcceptanceRate,
                                                              lowerLimitStepSize,
                                                              upperLimitStepSize,
                                                              iterationsToTestStepSize,
                                                              maxIterations});
                         }
                         catch (const std::runtime_error &e) {
                             EXPECT_STREQ(
                                     "Parameter error: lowerLimitAcceptanceRate is larger than upperLimitAcceptanceRate",
                                     e.what());
                             throw;
                         }
                     }, std::runtime_error);
    }

    TEST(AcceptanceRateTuner, throwsExceptionIfUpperLimitStepSizeIsLessThanLowerLimitStepSize) {
        double startingStepSize = 0.2;
        auto *markovChain
                = new hops::MarkovChainAdapter(
                        hops::StateRecorder(hops::MetropolisHastingsFilter(ProposerMock(startingStepSize))));
        hops::RandomNumberGenerator generator(42);

        float upperLimitAcceptanceRate = 0.3;
        float lowerLimitAcceptanceRate = 0.1;
        double lowerLimitStepSize = 1;
        double upperLimitStepSize = 0;
        size_t iterationsToTestStepSize = 100;
        size_t maxIterations = 1000;

        EXPECT_THROW({
                         try {
                             hops::AcceptanceRateTuner::tune(markovChain,
                                                             generator,
                                                             {lowerLimitAcceptanceRate,
                                                              upperLimitAcceptanceRate,
                                                              lowerLimitStepSize,
                                                              upperLimitStepSize,
                                                              iterationsToTestStepSize,
                                                              maxIterations});
                         }
                         catch (const std::runtime_error &e) {
                             EXPECT_STREQ(
                                     "Parameter error: lowerLimitStepSize is larger than upperLimitStepSize",
                                     e.what());
                             throw;
                         }
                     }, std::runtime_error);
    }

    TEST(AcceptanceRateTuner, throwsExceptionIfIterationsToTestStepSizeIs0) {
        double startingStepSize = 0.2;
        auto *markovChain
                = new hops::MarkovChainAdapter(
                        hops::StateRecorder(hops::MetropolisHastingsFilter(ProposerMock(startingStepSize))));
        hops::RandomNumberGenerator generator(42);

        float upperLimitAcceptanceRate = 0.3;
        float lowerLimitAcceptanceRate = 0.1;
        double lowerLimitStepSize = 0;
        double upperLimitStepSize = 1;
        size_t iterationsToTestStepSize = 0;
        size_t maxIterations = 1000;

        EXPECT_THROW({
                         try {
                             hops::AcceptanceRateTuner::tune(markovChain,
                                                             generator,
                                                             {lowerLimitAcceptanceRate,
                                                              upperLimitAcceptanceRate,
                                                              lowerLimitStepSize,
                                                              upperLimitStepSize,
                                                              iterationsToTestStepSize,
                                                              maxIterations});
                         }
                         catch (const std::runtime_error &e) {
                             EXPECT_STREQ(
                                     "Parameter error: iterationsToTestStepSize is 0",
                                     e.what());
                             throw;
                         }
                     }, std::runtime_error);
    }

    TEST(AcceptanceRateTuner, tuneByIncreasingStepSize) {
        double startingStepSize = 0.2;
        auto *markovChain
                = new hops::MarkovChainAdapter(
                        hops::StateRecorder(hops::MetropolisHastingsFilter(ProposerMock(startingStepSize))));
        hops::RandomNumberGenerator generator(42);

        float upperLimitAcceptanceRate = 0.3;
        float lowerLimitAcceptanceRate = 0.1;
        double lowerLimitStepSize = 0;
        double upperLimitStepSize = 1;
        size_t iterationsToTestStepSize = 100;
        size_t maxIterations = 1000;

        bool isTuned = hops::AcceptanceRateTuner::tune(markovChain,
                                                       generator,
                                                       {lowerLimitAcceptanceRate,
                                                        upperLimitAcceptanceRate,
                                                        lowerLimitStepSize,
                                                        upperLimitStepSize,
                                                        iterationsToTestStepSize,
                                                        maxIterations});

        auto actualAcceptanceRate = markovChain->getAcceptanceRate();
        EXPECT_TRUE(isTuned);
        EXPECT_LE(markovChain->getNumberOfStepsTaken(), maxIterations + iterationsToTestStepSize);
        EXPECT_LE(actualAcceptanceRate, upperLimitAcceptanceRate);
        EXPECT_GE(actualAcceptanceRate, lowerLimitAcceptanceRate);
    }

    TEST(AcceptanceRateTuner, tuneByDecreasingStepSize) {
        double startingStepSize = 0.2;
        auto *markovChain
                = new hops::MarkovChainAdapter(
                        hops::StateRecorder(hops::MetropolisHastingsFilter(ProposerMock(startingStepSize))));
        hops::RandomNumberGenerator generator(42);

        float upperLimitAcceptanceRate = 0.7;
        float lowerLimitAcceptanceRate = 0.6;
        double lowerLimitStepSize = 0;
        double upperLimitStepSize = 1;
        size_t iterationsToTestStepSize = 100;
        size_t maxIterations = 1000;

        bool isTuned = hops::AcceptanceRateTuner::tune(markovChain,
                                                       generator,
                                                       {lowerLimitAcceptanceRate,
                                                        upperLimitAcceptanceRate,
                                                        lowerLimitStepSize,
                                                        upperLimitStepSize,
                                                        iterationsToTestStepSize,
                                                        maxIterations});

        auto actualAcceptanceRate = markovChain->getAcceptanceRate();
        EXPECT_TRUE(isTuned);
        EXPECT_LE(markovChain->getNumberOfStepsTaken(), maxIterations + iterationsToTestStepSize);
        EXPECT_LE(actualAcceptanceRate, upperLimitAcceptanceRate);
        EXPECT_GE(actualAcceptanceRate, lowerLimitAcceptanceRate);
    }

    TEST(AcceptanceRateTuner, stopsWhenMaxIterationsAreReached) {
        double startingStepSize = 0.2;
        auto *markovChain
                = new hops::MarkovChainAdapter(
                        hops::StateRecorder(hops::MetropolisHastingsFilter(ProposerMock(startingStepSize))));
        hops::RandomNumberGenerator generator(42);

        float upperLimitAcceptanceRate = 0.7;
        float lowerLimitAcceptanceRate = 0.6;
        double lowerLimitStepSize = 0.6;
        double upperLimitStepSize = 1;
        size_t iterationsToTestStepSize = 100;
        size_t maxIterations = 1000;

        bool isTuned = hops::AcceptanceRateTuner::tune(markovChain,
                                                       generator,
                                                       {lowerLimitAcceptanceRate,
                                                        upperLimitAcceptanceRate,
                                                        lowerLimitStepSize,
                                                        upperLimitStepSize,
                                                        iterationsToTestStepSize,
                                                        maxIterations});

        EXPECT_FALSE(isTuned);
        EXPECT_LE(markovChain->getNumberOfStepsTaken(), maxIterations + iterationsToTestStepSize);
    }
}
