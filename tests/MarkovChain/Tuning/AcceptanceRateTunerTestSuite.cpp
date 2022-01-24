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

#include <hops/MarkovChain/Draw/MetropolisHastingsFilter.hpp>
#include <hops/MarkovChain/MarkovChain.hpp>
#include <hops/MarkovChain/MarkovChainAdapter.hpp>
#include <hops/MarkovChain/Proposal/Proposal.hpp>
#include <hops/MarkovChain/Tuning/AcceptanceRateTuner.hpp>

namespace {
    class ProposerMock : public hops::Proposal {
    public:
        explicit ProposerMock(double stepSize) : stepSize(stepSize) {}

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

        [[nodiscard]] std::string getProposalName() const override { return "ProposerMock"; };

        void setParameter(const hops::ProposalParameter &parameter, const std::any &value) override {
            stepSize = std::any_cast<double>(value);
        }

        long getNumberOfStepsTaken() const {
            return numberOfStepsTaken;
        }

        void setState(const hops::VectorType &) override {}

        hops::VectorType getProposal() const override { return hops::VectorType(); }

        std::vector<std::string> getParameterNames() const override { return std::vector<std::string>(); }

        std::any getParameter(const hops::ProposalParameter &parameter) const override {
            return std::any();
        }

        std::string getParameterType(const hops::ProposalParameter &parameter) const override {
            return std::string();
        }

        bool hasStepSize() const override {
            return true;
        }

        std::unique_ptr<Proposal> copyProposal() const override {
            return std::make_unique<ProposerMock>(*this);
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
        double lowerLimitStepSize = 1e-1;
        double upperLimitStepSize = 1;
        double smoothingLength = 1;
        size_t stepSizeGridSize = 21;
        size_t iterationsToTestStepSize = 500;
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
                true
        };

        Eigen::MatrixXd data;

        Eigen::VectorXd optimalParameter;
        double optimalValue;

        std::vector<std::shared_ptr<hops::MarkovChain>> mcs{markovChain};
        std::vector<hops::RandomNumberGenerator *> generators{&generator};
        bool isTuned = hops::AcceptanceRateTuner::tune(
                optimalParameter,
                optimalValue,
                mcs,
                generators,
                parameters,
                data
        );

        std::cout << data << std::endl;

        double actualAcceptanceRate = std::get<0>(markovChain->draw(generator, 5000));
        std::cout << optimalParameter << " " << actualAcceptanceRate << std::endl;
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
