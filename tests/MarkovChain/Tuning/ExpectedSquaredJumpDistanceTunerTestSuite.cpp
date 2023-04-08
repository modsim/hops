#define BOOST_TEST_DYN_LINK
#define BOOST_TEST_MODULE ExpectedSquaredJumpDistanceTunerTestSuite

#include <memory>
#include <boost/test/unit_test.hpp>
#include <cmath>
#include <vector>
#include <Eigen/Core>

#include "hops/MarkovChain/MarkovChain.hpp"
#include "hops/MarkovChain/MarkovChainAdapter.hpp"
#include "hops/MarkovChain/Draw/MetropolisHastingsFilter.hpp"
#include "hops/MarkovChain/Proposal/Proposal.hpp"
#include "hops/MarkovChain/Tuning/ExpectedSquaredJumpDistanceTuner.hpp"

namespace {
    class ProposalMock : public hops::Proposal {
    public:
        explicit ProposalMock(double stepSize, double targetStepSize) :
                stepSize(stepSize),
                targetStepSize(targetStepSize),
                x(hops::VectorType::Zero(1)),
                y(hops::VectorType::Zero(1)) {}

        hops::VectorType &propose(hops::RandomNumberGenerator &) override {
            numberOfStepsTaken++;
            y = x +
                std::exp(-std::pow(std::log10(targetStepSize) - std::log10(stepSize), 2)) * hops::VectorType::Ones(1);
            return y;
        }

        hops::VectorType &acceptProposal() override {
            x = y;
            return x;
        };

        [[nodiscard]] double computeLogAcceptanceProbability() override {
            return 0;
        };

        [[nodiscard]] hops::VectorType getState() const override { return x; };

        [[nodiscard]] std::string getProposalName() const override { return "ProposalMock"; };

        void setParameter(const hops::ProposalParameter &parameter, const std::any &value) override {
            stepSize = std::any_cast<double>(value);
        }

        void setState(const Eigen::VectorXd &) override {};

        void setDimensionNames(const std::vector<std::string> &names) override { }

        std::vector<std::string> getDimensionNames() const override {
            return std::vector<std::string>();
        }

        std::any getParameter(const hops::ProposalParameter &parameter) const override {
            return stepSize;
        }

        long getNumberOfStepsTaken() const {
            return numberOfStepsTaken;
        }

        hops::VectorType getProposal() const override {
            return hops::VectorType();
        }

        [[nodiscard]] std::vector<std::string> getParameterNames() const override {
            return std::vector<std::string>();
        }

        std::string getParameterType(const hops::ProposalParameter &parameter) const override {
            return "double";
        }

        [[nodiscard]] std::unique_ptr<Proposal> copyProposal() const override {
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
        double targetStepSize;
        long numberOfStepsTaken = 0;
        hops::VectorType x;
        hops::VectorType y;
    };
}

BOOST_AUTO_TEST_SUITE(ExpectedSquaredJumpDistanceTuner)

    BOOST_AUTO_TEST_CASE(nothingToTune) {
        double startingStepSize = 0.5;
        double targetStepSize = 1;
        auto markovChain
                = std::make_shared<
                        decltype(hops::MarkovChainAdapter(
                                hops::MetropolisHastingsFilter(ProposalMock(startingStepSize, targetStepSize))))>(
                        hops::MetropolisHastingsFilter(ProposalMock(startingStepSize, targetStepSize))
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
        std::vector<hops::RandomNumberGenerator *> generators{&generator};
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
        double expectedSquaredJumpDistance = hops::computeExpectedSquaredJumpDistance<Eigen::VectorXd, Eigen::MatrixXd>(
                states, sqrtCovariance, 1);

        BOOST_CHECK_EQUAL(optimalParameter(0), 1);
    }

BOOST_AUTO_TEST_SUITE_END()
