#define BOOST_TEST_DYN_LINK
#define BOOST_TEST_MODULE ModelMixinTestSuite

#include <boost/test/unit_test.hpp>
#include <memory>
#include <utility>

#include "hops/MarkovChain/ModelMixin.hpp"
#include "hops/MarkovChain/Proposal/Proposal.hpp"
#include "hops/Model/Model.hpp"
#include "hops/Utility/VectorType.hpp"

namespace {
    class ModelMock : public hops::Model {
    private:
        double internal_state =0;

    public:
        [[nodiscard]] double computeNegativeLogLikelihood(const Eigen::VectorXd &state) override {
            return state(0)+internal_state;
        }

        [[nodiscard]] std::vector<std::string> getDimensionNames() const override {
            return std::vector<std::string>{"correct name"};
        }

        [[nodiscard]] std::unique_ptr<Model> copyModel() const override {
            return std::make_unique<ModelMock>(*this);
        }
    };

    class ProposalMock : public hops::Proposal {
    public:
        [[nodiscard]] hops::VectorType getState() const override {
            return state;
        }

        [[nodiscard]] hops::VectorType getProposal() const override {
            return proposal;
        }

        void setProposal(const hops::VectorType &newProposal) override {
            ProposalMock::proposal = newProposal;
        }

        hops::VectorType &acceptProposal() override {
            state = proposal;
            return state;
        }

        void setDimensionNames(const std::vector<std::string> &names) override {
            dimensionName = names[0];
        }

        [[nodiscard]] std::vector<std::string> getDimensionNames() const override {
            return {dimensionName};
        }

        hops::VectorType &propose(hops::RandomNumberGenerator &) override {
            return proposal;
        }

        hops::VectorType &propose(hops::RandomNumberGenerator &rng, const Eigen::VectorXd &activeIndices) override {
            return Proposal::propose(rng, activeIndices);
        }

        double computeLogAcceptanceProbability() override {
            return state(0) - proposal(0);
        }

        void setState(const hops::VectorType &newState) override {
            ProposalMock::state = std::move(newState);
        }

        [[nodiscard]] std::vector<std::string> getParameterNames() const override {
            return std::vector<std::string>{};
        }

        std::any getParameter(const hops::ProposalParameter &) const override {
            return std::any(0);
        }

        std::string getParameterType(const hops::ProposalParameter &) const override {
            return std::string();
        }

        void setParameter(const hops::ProposalParameter &, const std::any &) override {
        }

        [[nodiscard]] std::string getProposalName() const override {
            return "MockProposal";
        }

        [[nodiscard]] std::unique_ptr<Proposal> copyProposal() const override {
            return std::make_unique<ProposalMock>(*this);
        }

        [[nodiscard]] const hops::MatrixType &getA() const override {
            throw std::runtime_error("Should not be called");
        }

        [[nodiscard]] const hops::VectorType &getB() const override {
            throw std::runtime_error("Should not be called");
        }

    private:
        hops::VectorType proposal;
        hops::VectorType state = Eigen::VectorXd::Ones(1);
        std::string dimensionName = "wrong name";
    };
}

BOOST_AUTO_TEST_SUITE(ModelMixin)

    BOOST_AUTO_TEST_CASE(testAcceptProposal) {
        auto model = ModelMock();
        hops::ModelMixin markovChainWithModelMixedIn((ProposalMock()), model);
        BOOST_CHECK(markovChainWithModelMixedIn.getStateNegativeLogLikelihood() == 1);

        markovChainWithModelMixedIn.acceptProposal();

        BOOST_CHECK(markovChainWithModelMixedIn.getStateNegativeLogLikelihood() == 0);
    }

    BOOST_AUTO_TEST_CASE(testCalculateLogAcceptanceProbabilityMultimodalModel) {
        ProposalMock markovChainMock;
        markovChainMock.setState(5 * Eigen::VectorXd::Ones(1));
        markovChainMock.setProposal(2 * Eigen::VectorXd::Ones(1));
        auto model = ModelMock();
        hops::ModelMixin markovChainWithModelMixedIn(markovChainMock, model);

        constexpr const double expectedValue = 2 * (5 - 2);
        BOOST_CHECK(markovChainWithModelMixedIn.computeLogAcceptanceProbability() == expectedValue);
    }

    BOOST_AUTO_TEST_CASE(testGetNegativeLogLikelihoodOfCurrentStateMultimodalModel) {
        ProposalMock markovChainMock;
        markovChainMock.setState(-5 * Eigen::VectorXd::Ones(1));
        auto model = ModelMock();
        hops::ModelMixin markovChainWithModelMixedIn(markovChainMock, model);
        BOOST_CHECK(markovChainWithModelMixedIn.getStateNegativeLogLikelihood() == -5);
    }

    BOOST_AUTO_TEST_CASE(testGetDimensionNamesReturnsNamesFromModel) {
        ProposalMock markovChainMock;
        auto model = ModelMock();
        hops::ModelMixin markovChainWithModelMixedIn(markovChainMock, model);
        BOOST_CHECK(markovChainWithModelMixedIn.getDimensionNames() == model.getDimensionNames());
        markovChainWithModelMixedIn.setDimensionNames({"wrong name"});
        BOOST_CHECK(markovChainWithModelMixedIn.getDimensionNames() != model.getDimensionNames());
    }

BOOST_AUTO_TEST_SUITE_END()
