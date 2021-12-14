#define BOOST_TEST_MODULE ExpectedSquaredJumpDistanceTunerTestSuite
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
        using StateType = Eigen::VectorXd;

        explicit ProposerMock(double stepSize, double targetStepSize) : 
                stepSize(stepSize), 
                targetStepSize(targetStepSize), 
                x(StateType::Zero(1)),
                y(StateType::Zero(1)) {}

        StateType propose(hops::RandomNumberGenerator) { numberOfStepsTaken++;
            y = x + std::exp(-std::pow(std::log10(targetStepSize) - std::log10(stepSize), 2)) * StateType::Ones(1);
            return y;
        }

        StateType acceptProposal() {
            x = y;
            return x;
        };

        [[nodiscard]] double computeLogAcceptanceProbability() const {
            return 0;
        };

        const std::vector<Eigen::VectorXd> &getStateRecords() {
            throw std::runtime_error("no records");
        }

        [[nodiscard]] StateType getState() const { return x; };

        [[nodiscard]] double getStateNegativeLogLikelihood() const { return 0; };

        [[nodiscard]] std::string getProposalName() const { return "ProposerMock"; };

        void setParameter(hops::ProposalParameter parameter, std::any value) {
            stepSize = std::any_cast<double>(value);
        }

        void setState(Eigen::VectorXd) {};

        std::any getParameter(hops::ProposalParameter parameter) const {
            return stepSize;
        }

        long getNumberOfStepsTaken() const {
            return numberOfStepsTaken;
        }

    private:
        double stepSize;
        double targetStepSize;
        long numberOfStepsTaken = 0;
        StateType x;
        StateType y;
    };
}

BOOST_AUTO_TEST_SUITE(ExpectedSquaredJumpDistanceTuner)

    BOOST_AUTO_TEST_CASE(nothingToTune) {
        double startingStepSize = 0.5;
        double targetStepSize = 1;
        auto markovChain
                = std::make_shared<
                        decltype(hops::MarkovChainAdapter(
                                hops::MetropolisHastingsFilter(ProposerMock(startingStepSize, targetStepSize))))>(
                        hops::MetropolisHastingsFilter(ProposerMock(startingStepSize, targetStepSize))
                );
        hops::RandomNumberGenerator generator(42);
        markovChain->setParameter(hops::ProposalParameter::STEP_SIZE, startingStepSize);

        double lowerLimitStepSize = 1e-2;
        double upperLimitStepSize = 1e2;
        double smoothingLength = 1;
        size_t stepSizeGridSize = 21;
        size_t iterationsToTestStepSize = 300;
        size_t maxPosteriorUpdates = 20;
        size_t maxPureSamplingRounds = 1;
        size_t iterationsForConvergence = 20;

        hops::ExpectedSquaredJumpDistanceTuner::param_type parameters{
                iterationsToTestStepSize,
                maxPosteriorUpdates,
                maxPureSamplingRounds,
                iterationsForConvergence,
                stepSizeGridSize,
                lowerLimitStepSize,
                upperLimitStepSize,
                smoothingLength,
                123,
                true,
                {1},
                false,
                false
        };

        Eigen::MatrixXd data;

        Eigen::VectorXd optimalParameter;
        double optimalValue;

        std::vector<std::shared_ptr<hops::MarkovChain>> mcs{markovChain};
        std::vector<hops::RandomNumberGenerator*> generators{&generator};
        bool isTuned = hops::ExpectedSquaredJumpDistanceTuner::tune(
                optimalParameter,
                optimalValue,
                mcs,
                generators,
                parameters,
                data
        );

        std::cout << data << std::endl;

        std::vector<Eigen::VectorXd> states(5000);
        for (size_t j = 0; j < 5000; ++j) {
            states[j] = std::get<1>(markovChain->draw(generator));
        }

        Eigen::MatrixXd sqrtCovariance = Eigen::MatrixXd::Identity(1, 1);
        double expectedSquaredJumpDistance = hops::computeExpectedSquaredJumpDistance<Eigen::VectorXd, Eigen::MatrixXd>(states, sqrtCovariance, 1);

        BOOST_CHECK_EQUAL(optimalParameter(0), 1);
    }

BOOST_AUTO_TEST_SUITE_END()
