#include "hops/MarkovChain/Proposal/Proposal.hpp"
#include "hops/MarkovChain/Proposal/ProposalParameter.hpp"
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

        StateType propose(hops::RandomNumberGenerator) { numberOfStepsTaken++;
            return getState();
        }

        StateType acceptProposal() {
            return getState();
        };

        [[nodiscard]] double computeLogAcceptanceProbability() const {
            return std::log(1 - stepSize);
        };

        const std::vector<Eigen::VectorXd> &getStateRecords() {
            throw std::runtime_error("no records");
        }

        [[nodiscard]] StateType getState() const { return Eigen::VectorXd::Zero(1); };

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
        long numberOfStepsTaken = 0;
    };
}

BOOST_AUTO_TEST_SUITE(AcceptanceRateTuner)

    BOOST_AUTO_TEST_CASE(nothingToTune) {
        double startingStepSize = 0.5;
        auto markovChain
                = std::make_shared<
                        decltype(hops::MarkovChainAdapter(
                                hops::MetropolisHastingsFilter(ProposerMock(startingStepSize))))>(
                        hops::MetropolisHastingsFilter(ProposerMock(startingStepSize))
                );
        hops::RandomNumberGenerator generator(42);
        markovChain->setParameter(hops::ProposalParameter::STEP_SIZE, startingStepSize);

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


        double actualAcceptanceRate = std::get<0>(markovChain->draw(generator, 5000));
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
