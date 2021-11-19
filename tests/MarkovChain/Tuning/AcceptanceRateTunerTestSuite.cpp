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

        std::pair<double, StateType> propose(hops::RandomNumberGenerator) {
            numberOfStepsTaken++;
            return std::make_pair(std::log(1 - stepSize), Eigen::VectorXd::Zero(1));
        }

        void acceptProposal() {};

        [[nodiscard]] double computeLogAcceptanceProbability() const {
            return std::log(1 - stepSize);
        };

        const std::vector<Eigen::VectorXd> &getStateRecords() {
            throw 0;
        }

        [[nodiscard]] StateType getState() const { return Eigen::VectorXd::Zero(1); };

        [[nodiscard]] std::string getProposalName() const { return "ProposerMock"; };

        void setStepSize(double newStepSize) {
            stepSize = newStepSize;
        }

        void setState(Eigen::VectorXd) {};

        [[nodiscard]] std::optional<double> getStepSize() const {
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
        double smoothingLength = 1;
        size_t stepSizeGridSize = 21;
        size_t iterationsToTestStepSize = 200;
        size_t maxPosteriorUpdates = 20;
        size_t maxPureSamplingRounds = 1;
        size_t iterationsForConvergence = 5;

        hops::AcceptanceRateTuner::param_type parameters{
                targetAcceptanceRate,
                iterationsToTestStepSize,
                maxPosteriorUpdates,
                maxPureSamplingRounds,
                iterationsForConvergence,
                stepSizeGridSize,
                lowerLimitStepSize,
                upperLimitStepSize,
                smoothingLength,
                42,
                false
        };

        std::vector<std::shared_ptr<hops::MarkovChain>> mcs{markovChain};
        std::vector<hops::RandomNumberGenerator> generators{generator};
        bool isTuned = hops::AcceptanceRateTuner::tune(
                mcs,
                generators,
                parameters
        );


        markovChain->draw(generator, 5000);
        double actualAcceptanceRate = markovChain->getAcceptanceRate();
        std::cout << actualAcceptanceRate << std::endl;
        BOOST_CHECK(isTuned);
        BOOST_CHECK_LE(markovChain->getNumberOfStepsTaken(),
                       maxPosteriorUpdates *
                               iterationsToTestStepSize + 5000);
        double upperLimitAcceptanceRate = 0.85;
        double lowerLimitAcceptanceRate = 0.75;
        BOOST_CHECK_LE(actualAcceptanceRate,
                       upperLimitAcceptanceRate);
        BOOST_CHECK_GE(actualAcceptanceRate,
                       lowerLimitAcceptanceRate);
    }

BOOST_AUTO_TEST_SUITE_END()
