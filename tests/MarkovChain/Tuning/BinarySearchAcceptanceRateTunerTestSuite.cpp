#include "hops/MarkovChain/MarkovChain.hpp"
#include "hops/MarkovChain/Proposal/Proposal.hpp"

#define BOOST_TEST_MODULE AcceptanceRateTunerTestSuite
#define BOOST_TEST_DYN_LINK

#include <boost/test/included/unit_test.hpp>
#include <cmath>
#include <Eigen/Core>

#include <hops/MarkovChain/MarkovChainAdapter.hpp>
#include <hops/MarkovChain/Draw/MetropolisHastingsFilter.hpp>
#include <hops/MarkovChain/Tuning/BinarySearchAcceptanceRateTuner.hpp>

namespace {
    class ProposalMock : public hops::Proposal {
    public:
        explicit ProposalMock(double stepSize) : stepSize(stepSize) {}

        hops::VectorType &propose(hops::RandomNumberGenerator &) override {
            numberOfStepsTaken++;
            return state;
        }

        hops::VectorType &acceptProposal() override {
            return state;
        };

        [[nodiscard]] double computeLogAcceptanceProbability() override {
            return std::log(1 - stepSize);
        };

        [[nodiscard]] hops::VectorType getState() const override { return state; };

        [[nodiscard]] std::string getProposalName() const override { return "ProposalMock"; };

        void setParameter(const hops::ProposalParameter &parameter, const std::any &value) override {
            stepSize = std::any_cast<double>(value);
        }

        void setState(const Eigen::VectorXd &) override {};

        std::any getParameter(const hops::ProposalParameter &parameter) const override {
            return stepSize;
        }

        long getNumberOfStepsTaken() const {
            return numberOfStepsTaken;
        }

        hops::VectorType getProposal() const override {
            return hops::VectorType();
        }

        std::vector<std::string> getParameterNames() const override {
            return std::vector<std::string>();
        }

        std::string getParameterType(const hops::ProposalParameter &parameter) const override {
            return std::string();
        }

        std::unique_ptr<Proposal> copyProposal() const override {
            return std::make_unique<ProposalMock>(*this);
        }

        const hops::MatrixType &getA() const override {
            throw std::runtime_error("Should not be called");
        }

        const hops::VectorType &getB() const override {
            throw std::runtime_error("Should not be called");
        }

    private:
        double stepSize;
        long numberOfStepsTaken = 0;

        hops::VectorType state = Eigen::VectorXd::Zero(1);
    };
}

BOOST_AUTO_TEST_SUITE(BinarySearchAcceptanceRateTuner)

    BOOST_AUTO_TEST_CASE(nothingToTune) {
        double startingStepSize = 0.2;
        hops::MarkovChain *markovChain
                = new hops::MarkovChainAdapter(hops::MetropolisHastingsFilter(ProposalMock(startingStepSize)));
        hops::RandomNumberGenerator generator(42);

        double upperLimitAcceptanceRate = 0.85;
        double lowerLimitAcceptanceRate = 0.8;
        double lowerLimitStepSize = 0;
        double upperLimitStepSize = 1;
        size_t iterationsToTestStepSize = 100;
        size_t maxIterations = 1000;

        double actualAcceptanceRate = -1;
        bool isTuned = hops::BinarySearchAcceptanceRateTuner::tune(startingStepSize,
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
        //BOOST_CHECK_LE(markovChain->getNumberOfStepsTaken(), maxIterations + iterationsToTestStepSize);
        BOOST_CHECK_LE(actualAcceptanceRate, upperLimitAcceptanceRate);
        BOOST_CHECK_GE(actualAcceptanceRate, lowerLimitAcceptanceRate);

        delete markovChain;
    }

    BOOST_AUTO_TEST_CASE(throwsExceptionIfUpperLimitAcceptanceRateIsLessThanLowerLimitAcceptanceRate) {
        double startingStepSize = 0.2;
        hops::MarkovChain *markovChain
                = new hops::MarkovChainAdapter(hops::MetropolisHastingsFilter(ProposalMock(startingStepSize)));
        hops::RandomNumberGenerator generator(42);

        double upperLimitAcceptanceRate = 0;
        double lowerLimitAcceptanceRate = 1;
        double lowerLimitStepSize = 0;
        double upperLimitStepSize = 1;
        size_t iterationsToTestStepSize = 100;
        size_t maxIterations = 1000;

        BOOST_CHECK_EXCEPTION(hops::BinarySearchAcceptanceRateTuner::tune(markovChain,
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
        hops::MarkovChain *markovChain
                = new hops::MarkovChainAdapter(hops::MetropolisHastingsFilter(ProposalMock(startingStepSize)));
        hops::RandomNumberGenerator generator(42);

        float upperLimitAcceptanceRate = 0.3;
        float lowerLimitAcceptanceRate = 0.1;
        double lowerLimitStepSize = 1;
        double upperLimitStepSize = 0;
        size_t iterationsToTestStepSize = 100;
        size_t maxIterations = 1000;

        BOOST_CHECK_EXCEPTION(hops::BinarySearchAcceptanceRateTuner::tune(
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
        hops::MarkovChain *markovChain
                = new hops::MarkovChainAdapter(hops::MetropolisHastingsFilter(ProposalMock(startingStepSize)));
        hops::RandomNumberGenerator generator(42);

        float upperLimitAcceptanceRate = 0.3;
        float lowerLimitAcceptanceRate = 0.1;
        double lowerLimitStepSize = 0;
        double upperLimitStepSize = 1;
        size_t iterationsToTestStepSize = 0;
        size_t maxIterations = 1000;

        BOOST_CHECK_EXCEPTION(hops::BinarySearchAcceptanceRateTuner::tune(
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
        hops::MarkovChain *markovChain
                = new hops::MarkovChainAdapter(hops::MetropolisHastingsFilter(ProposalMock(startingStepSize)));
        hops::RandomNumberGenerator generator(42);

        float upperLimitAcceptanceRate = 0.3;
        float lowerLimitAcceptanceRate = 0.1;
        double lowerLimitStepSize = 0;
        double upperLimitStepSize = 1;
        size_t iterationsToTestStepSize = 100;
        size_t maxIterations = 1000;

        double actualAcceptanceRate = -1;
        bool isTuned = hops::BinarySearchAcceptanceRateTuner::tune(startingStepSize,
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
        //BOOST_CHECK_LE(markovChain->getNumberOfStepsTaken(), maxIterations + iterationsToTestStepSize);
        BOOST_CHECK_LE(actualAcceptanceRate, upperLimitAcceptanceRate);
        BOOST_CHECK_GE(actualAcceptanceRate, lowerLimitAcceptanceRate);

        delete markovChain;
    }

    BOOST_AUTO_TEST_CASE(tuneByDecreasingStepSize) {
        double startingStepSize = 0.2;
        hops::MarkovChain *markovChain
                = new hops::MarkovChainAdapter(hops::MetropolisHastingsFilter(ProposalMock(startingStepSize)));
        hops::RandomNumberGenerator generator(42);

        float upperLimitAcceptanceRate = 0.7;
        float lowerLimitAcceptanceRate = 0.6;
        double lowerLimitStepSize = 0;
        double upperLimitStepSize = 1;
        size_t iterationsToTestStepSize = 100;
        size_t maxIterations = 1000;

        double actualAcceptanceRate = -1;
        bool isTuned = hops::BinarySearchAcceptanceRateTuner::tune(startingStepSize,
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
        //BOOST_CHECK_LE(markovChain->getNumberOfStepsTaken(), maxIterations + iterationsToTestStepSize);
        BOOST_CHECK_LE(actualAcceptanceRate, upperLimitAcceptanceRate);
        BOOST_CHECK_GE(actualAcceptanceRate, lowerLimitAcceptanceRate);

        delete markovChain;
    }

    BOOST_AUTO_TEST_CASE(stopsWhenMaxIterationsAreReached) {
        double startingStepSize = 0.2;
        hops::MarkovChain *markovChain
                = new hops::MarkovChainAdapter(hops::MetropolisHastingsFilter(ProposalMock(startingStepSize)));
        hops::RandomNumberGenerator generator(42);

        float upperLimitAcceptanceRate = 0.7;
        float lowerLimitAcceptanceRate = 0.6;
        double lowerLimitStepSize = 0.6;
        double upperLimitStepSize = 1;
        size_t iterationsToTestStepSize = 100;
        size_t maxIterations = 1000;

        bool isNotTuned = !hops::BinarySearchAcceptanceRateTuner::tune(markovChain,
                                                                       generator,
                                                                       {lowerLimitAcceptanceRate,
                                                                        upperLimitAcceptanceRate,
                                                                        lowerLimitStepSize,
                                                                        upperLimitStepSize,
                                                                        iterationsToTestStepSize,
                                                                        maxIterations});

        BOOST_CHECK(isNotTuned);
        //BOOST_CHECK_LE(markovChain->getNumberOfStepsTaken(), maxIterations + iterationsToTestStepSize);

        delete markovChain;
    }

BOOST_AUTO_TEST_SUITE_END()
