#define BOOST_TEST_MODULE AcceptanceRateTunerTestSuite
#define BOOST_TEST_DYN_LINK

#include <boost/test/included/unit_test.hpp>
#include <cmath>
#include <Eigen/Core>
#include <hops/hops.hpp>

namespace {
    class ProposerMock {
    public:
        explicit ProposerMock(double stepSize) : stepSize(stepSize) {}

        using StateType = Eigen::VectorXd;

        void propose(hops::RandomNumberGenerator) { numberOfStepsTaken++; }

        void acceptProposal() {};

        [[nodiscard]] double calculateLogAcceptanceProbability() const {
            return std::log(1 - stepSize);
        };

        const std::vector<Eigen::VectorXd> &getStateRecords() {
            throw 0;
        }

        [[nodiscard]] StateType getState() const { return Eigen::VectorXd::Zero(1); };

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
}

BOOST_AUTO_TEST_SUITE(AcceptanceRateTuner)

    BOOST_AUTO_TEST_CASE(nothingToTune) {
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

        double actualAcceptanceRate = -1;
        bool isTuned = hops::AcceptanceRateTuner::tune(startingStepSize,
                                                       actualAcceptanceRate,
                                                       markovChain,
                                                       generator,
                                                       {lowerLimitAcceptanceRate,
                                                        upperLimitAcceptanceRate,
                                                        lowerLimitStepSize,
                                                        upperLimitStepSize,
                                                        iterationsToTestStepSize,
                                                        maxIterations});


        BOOST_CHECK(isTuned);
        BOOST_CHECK_LE(markovChain->getNumberOfStepsTaken(), maxIterations + iterationsToTestStepSize);
        BOOST_CHECK_LE(actualAcceptanceRate, upperLimitAcceptanceRate);
        BOOST_CHECK_GE(actualAcceptanceRate, lowerLimitAcceptanceRate);

        delete markovChain;
    }

    BOOST_AUTO_TEST_CASE(throwsExceptionIfUpperLimitAcceptanceRateIsLessThanLowerLimitAcceptanceRate) {
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

        BOOST_CHECK_EXCEPTION(hops::AcceptanceRateTuner::tune(markovChain,
                                                              generator,
                                                              {lowerLimitAcceptanceRate,
                                                               upperLimitAcceptanceRate,
                                                               lowerLimitStepSize,
                                                               upperLimitStepSize,
                                                               iterationsToTestStepSize,
                                                               maxIterations}),
                              std::runtime_error,
                              [](auto ex) {
                                  return std::string(ex.what()) ==
                                         "Parameter error: lowerLimitAcceptanceRate is larger than upperLimitAcceptanceRate";
                              });

        delete markovChain;
    }

    BOOST_AUTO_TEST_CASE(throwsExceptionIfUpperLimitStepSizeIsLessThanLowerLimitStepSize) {
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

        BOOST_CHECK_EXCEPTION(hops::AcceptanceRateTuner::tune(
                markovChain,
                generator,
                {lowerLimitAcceptanceRate,
                 upperLimitAcceptanceRate,
                 lowerLimitStepSize,
                 upperLimitStepSize,
                 iterationsToTestStepSize,
                 maxIterations}),
                              std::runtime_error,
                              [](auto ex) {
                                  return std::string(ex.what()) ==
                                         "Parameter error: lowerLimitStepSize is larger than upperLimitStepSize";
                              });

        delete markovChain;
    }

    BOOST_AUTO_TEST_CASE(throwsExceptionIfIterationsToTestStepSizeIs0) {
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

        BOOST_CHECK_EXCEPTION(hops::AcceptanceRateTuner::tune(
                markovChain,
                generator,
                {lowerLimitAcceptanceRate,
                 upperLimitAcceptanceRate,
                 lowerLimitStepSize,
                 upperLimitStepSize,
                 iterationsToTestStepSize,
                 maxIterations}),
                              std::runtime_error,
                              [](auto ex) {
                                  return std::string(ex.what()) ==
                                         "Parameter error: iterationsToTestStepSize is 0";
                              });
        delete markovChain;
    }

    BOOST_AUTO_TEST_CASE(tuneByIncreasingStepSize) {
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

        double actualAcceptanceRate = -1;
        bool isTuned = hops::AcceptanceRateTuner::tune(startingStepSize,
                                                       actualAcceptanceRate,
                                                       markovChain,
                                                       generator,
                                                       {lowerLimitAcceptanceRate,
                                                        upperLimitAcceptanceRate,
                                                        lowerLimitStepSize,
                                                        upperLimitStepSize,
                                                        iterationsToTestStepSize,
                                                        maxIterations});

        BOOST_CHECK(isTuned);
        BOOST_CHECK_LE(markovChain->getNumberOfStepsTaken(), maxIterations + iterationsToTestStepSize);
        BOOST_CHECK_LE(actualAcceptanceRate, upperLimitAcceptanceRate);
        BOOST_CHECK_GE(actualAcceptanceRate, lowerLimitAcceptanceRate);

        delete markovChain;
    }

    BOOST_AUTO_TEST_CASE(tuneByDecreasingStepSize) {
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

        double actualAcceptanceRate = -1;
        bool isTuned = hops::AcceptanceRateTuner::tune(startingStepSize,
                                                       actualAcceptanceRate,
                                                       markovChain,
                                                       generator,
                                                       {lowerLimitAcceptanceRate,
                                                        upperLimitAcceptanceRate,
                                                        lowerLimitStepSize,
                                                        upperLimitStepSize,
                                                        iterationsToTestStepSize,
                                                        maxIterations});

        BOOST_CHECK(isTuned);
        BOOST_CHECK_LE(markovChain->getNumberOfStepsTaken(), maxIterations + iterationsToTestStepSize);
        BOOST_CHECK_LE(actualAcceptanceRate, upperLimitAcceptanceRate);
        BOOST_CHECK_GE(actualAcceptanceRate, lowerLimitAcceptanceRate);

        delete markovChain;
    }

    BOOST_AUTO_TEST_CASE(stopsWhenMaxIterationsAreReached) {
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

        bool isNotTuned = !hops::AcceptanceRateTuner::tune(markovChain,
                                                           generator,
                                                           {lowerLimitAcceptanceRate,
                                                            upperLimitAcceptanceRate,
                                                            lowerLimitStepSize,
                                                            upperLimitStepSize,
                                                            iterationsToTestStepSize,
                                                            maxIterations});

        BOOST_CHECK(isNotTuned);
        BOOST_CHECK_LE(markovChain->getNumberOfStepsTaken(), maxIterations + iterationsToTestStepSize);

        delete markovChain;
    }

BOOST_AUTO_TEST_SUITE_END()
