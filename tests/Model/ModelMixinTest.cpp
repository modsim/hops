#include <gtest/gtest.h>
#include <hops/Model/ModelMixin.hpp>

namespace {

    class ModelMock {
    public:
#pragma clang diagnostic push
#pragma ide diagnostic ignored "readability-convert-member-functions-to-static"

        double calculateNegativeLogLikelihood(double state) {
            return state;
        }

#pragma clang diagnostic pop
    };

    class MarkovChainMock {
    public:
        using StateType = double;

        StateType getProposal() const {
            return proposal;
        }

        void setProposal(StateType newProposal) {
            MarkovChainMock::proposal = newProposal;
        }

        StateType getState() const {
            return state;
        }

        void setState(StateType newState) {
            MarkovChainMock::state = newState;
        }

        void acceptProposal() {
            state = proposal;
        }

        double calculateLogAcceptanceProbability() {
            return state - proposal;
        }

    private:
        StateType proposal;
        StateType state = 1;
    };

    TEST(ModelMixin, testAcceptProposal) {
        hops::ModelMixin markovChainWithModelMixedIn((MarkovChainMock()), (ModelMock()));
        EXPECT_EQ(markovChainWithModelMixedIn.getNegativeLogLikelihoodOfCurrentState(), 1);

        markovChainWithModelMixedIn.acceptProposal();

        EXPECT_EQ(markovChainWithModelMixedIn.getNegativeLogLikelihoodOfCurrentState(), 0);
    }

    TEST(MultimodalModel, testCalculateLogAcceptanceProbability) {
        MarkovChainMock markovChainMock;
        markovChainMock.setState(5);
        markovChainMock.setProposal(2);
        hops::ModelMixin markovChainWithModelMixedIn(markovChainMock, (ModelMock()));

        constexpr const double expectedValue = 2 * (5 - 2);
        EXPECT_EQ(markovChainWithModelMixedIn.calculateLogAcceptanceProbability(), expectedValue);
    }

    TEST(MultimodalModel, testGetNegativeLogLikelihoodOfCurrentState) {
        MarkovChainMock markovChainMock;
        markovChainMock.setState(-5);
        hops::ModelMixin markovChainWithModelMixedIn(markovChainMock, (ModelMock()));
        EXPECT_EQ(markovChainWithModelMixedIn.getNegativeLogLikelihoodOfCurrentState(), -5);
    }
}
