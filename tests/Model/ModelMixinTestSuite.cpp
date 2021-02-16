#define BOOST_TEST_MODULE ModelMixinTestSuite
#define BOOST_TEST_DYN_LINK

#include <boost/test/included/unit_test.hpp>
#include <hops/Model/ModelMixin.hpp>

namespace {

    class ModelMock {
    public:
        static double calculateNegativeLogLikelihood(double state) {
            return state;
        }

    };

    class MarkovChainMock {
    public:
        using StateType = double;

        [[nodiscard]] StateType getProposal() const {
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

        double calculateLogAcceptanceProbability() const {
            return state - proposal;
        }

    private:
        StateType proposal{};
        StateType state = 1;
    };
}

BOOST_AUTO_TEST_SUITE(ModelMixin)

    BOOST_AUTO_TEST_CASE(testAcceptProposal) {
        hops::ModelMixin markovChainWithModelMixedIn((MarkovChainMock()), (ModelMock()));
        BOOST_CHECK(markovChainWithModelMixedIn.getNegativeLogLikelihoodOfCurrentState() == 1);

        markovChainWithModelMixedIn.acceptProposal();

        BOOST_CHECK(markovChainWithModelMixedIn.getNegativeLogLikelihoodOfCurrentState() == 0);
    }

    BOOST_AUTO_TEST_CASE(testCalculateLogAcceptanceProbabilityMultimodalModel) {
    MarkovChainMock markovChainMock;
    markovChainMock.setState(5);
    markovChainMock.setProposal(2);
    hops::ModelMixin markovChainWithModelMixedIn(markovChainMock, (ModelMock()));

        constexpr const double expectedValue = 2 * (5 - 2);
        BOOST_CHECK(markovChainWithModelMixedIn.calculateLogAcceptanceProbability() == expectedValue);
    }

    BOOST_AUTO_TEST_CASE(testGetNegativeLogLikelihoodOfCurrentStateMultimodalModel) {
        MarkovChainMock markovChainMock;
        markovChainMock.setState(-5);
        hops::ModelMixin markovChainWithModelMixedIn(markovChainMock, (ModelMock()));
        BOOST_CHECK(markovChainWithModelMixedIn.getNegativeLogLikelihoodOfCurrentState() == -5);
    }

BOOST_AUTO_TEST_SUITE_END()
