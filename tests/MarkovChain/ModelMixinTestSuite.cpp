#define BOOST_TEST_MODULE ModelMixinTestSuite
#define BOOST_TEST_DYN_LINK

#include <boost/test/included/unit_test.hpp>
#include <memory>

#include <hops/Model/Model.hpp>
#include <hops/MarkovChain/ModelMixin.hpp>
#include <utility>

namespace {
    class ModelMock : public hops::Model {
    public:
#pragma clang diagnostic push
#pragma ide diagnostic ignored "readability-convert-member-functions-to-static"
    [[nodiscard]] hops::MatrixType::Scalar computeNegativeLogLikelihood(const Eigen::VectorXd &state) const override {
            return state(0);
        }
#pragma clang diagnostic pop

        std::unique_ptr<Model> deepCopy() const override {
            return std::make_unique<ModelMock>();
        }
    };

    class MarkovChainMock {
    public:
        [[nodiscard]] hops::VectorType getProposal() const {
            return proposal;
        }

        void setProposal(hops::VectorType newProposal) {
            MarkovChainMock::proposal = std::move(newProposal);
        }

        [[nodiscard]] hops::VectorType getState() const {
            return state;
        }

        void setState(hops::VectorType newState) {
            MarkovChainMock::state = std::move(newState);
        }

        void acceptProposal() {
            state = proposal;
        }

        [[nodiscard]] double computeLogAcceptanceProbability() const {
            return state(0) - proposal(0);
        }

    private:
        hops::VectorType proposal;
        hops::VectorType state = Eigen::VectorXd::Ones(1);
    };
}

BOOST_AUTO_TEST_SUITE(ModelMixin)

    BOOST_AUTO_TEST_CASE(testAcceptProposal) {
        auto model = ModelMock();
        hops::ModelMixin markovChainWithModelMixedIn((MarkovChainMock()), model);
        BOOST_CHECK(markovChainWithModelMixedIn.getNegativeLogLikelihoodOfCurrentState() == 1);

        markovChainWithModelMixedIn.acceptProposal();

        BOOST_CHECK(markovChainWithModelMixedIn.getNegativeLogLikelihoodOfCurrentState() == 0);
    }

    BOOST_AUTO_TEST_CASE(testCalculateLogAcceptanceProbabilityMultimodalModel) {
        MarkovChainMock markovChainMock;
        markovChainMock.setState(5*Eigen::VectorXd::Ones(1));
        markovChainMock.setProposal(2*Eigen::VectorXd::Ones(1));
        auto model = ModelMock();
        hops::ModelMixin markovChainWithModelMixedIn(markovChainMock, model);

        constexpr const double expectedValue = 2 * (5 - 2);
        BOOST_CHECK(markovChainWithModelMixedIn.computeLogAcceptanceProbability() == expectedValue);
    }

    BOOST_AUTO_TEST_CASE(testGetNegativeLogLikelihoodOfCurrentStateMultimodalModel) {
        MarkovChainMock markovChainMock;
        markovChainMock.setState(-5*Eigen::VectorXd::Ones(1));
        auto model = ModelMock();
        hops::ModelMixin markovChainWithModelMixedIn(markovChainMock, model);
        BOOST_CHECK(markovChainWithModelMixedIn.getNegativeLogLikelihoodOfCurrentState() == -5);
    }

BOOST_AUTO_TEST_SUITE_END()
