#define BOOST_TEST_MODULE AcceptanceRateTunerTestSuite
#define BOOST_TEST_DYN_LINK

#include <memory>
#include <boost/test/included/unit_test.hpp>
#include <cmath>
#include <memory>
#include <vector>
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

        void setState(Eigen::VectorXd) {};

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
        double startingStepSize = 0.5;
        auto markovChain
                = std::make_shared<
                        decltype(hops::MarkovChainAdapter(
                                hops::StateRecorder(hops::MetropolisHastingsFilter(ProposerMock(startingStepSize)))))>(
                        hops::StateRecorder(hops::MetropolisHastingsFilter(ProposerMock(startingStepSize)))
                );
        hops::RandomNumberGenerator generator(42);
        markovChain->setStepSize(startingStepSize);

        double targetAcceptanceRate = 0.825;
        double lowerLimitStepSize = 1e-2;
        double upperLimitStepSize = 1;
        size_t iterationsToTestStepSize = 50;
        size_t maxIterations = 100;

        std::vector<std::shared_ptr<hops::MarkovChain>> mcs{markovChain};
        std::vector<hops::RandomNumberGenerator> generators{generator};
        bool isTuned = hops::AcceptanceRateTuner::tune(
                mcs,
                generators,
                {
                        targetAcceptanceRate,
                        iterationsToTestStepSize,
                        maxIterations,
                        200,
                        lowerLimitStepSize,
                        upperLimitStepSize,
                        42,
                        "test_output"
                });


        markovChain->draw(generator, 5000);
        double actualAcceptanceRate = markovChain->getAcceptanceRate();
        std::cout<< actualAcceptanceRate << std::endl;
        BOOST_CHECK(isTuned);
        BOOST_CHECK_LE(markovChain->getNumberOfStepsTaken(),
                       maxIterations *
                       iterationsToTestStepSize + 5000);
        double upperLimitAcceptanceRate = 0.85;
        double lowerLimitAcceptanceRate = 0.75;
        BOOST_CHECK_LE(actualAcceptanceRate,
                       upperLimitAcceptanceRate);
        BOOST_CHECK_GE(actualAcceptanceRate,
                       lowerLimitAcceptanceRate);
    }

BOOST_AUTO_TEST_SUITE_END()
