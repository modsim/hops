#define BOOST_TEST_MODULE ProposalFactoryTestSuite
#define BOOST_TEST_DYN_LINK

#include <boost/test/included/unit_test.hpp>
#include <Eigen/Core>

#include "hops/hops.hpp"

namespace {
    class ModelMock : public hops::Model {
    public:
        [[nodiscard]] double computeNegativeLogLikelihood(const hops::VectorType &x) const override {
            return 0;
        }
    };
}

struct ProposalFactoryTestFixture {
public:
//    ProposalFactoryTestFixture() {
//        A = Eigen::MatrixXd(4, 2);
//        A << Eigen::MatrixXd::Identity(2, 2), -Eigen::MatrixXd::Identity(2, 2);
//        b = Eigen::VectorXd::Ones(4);
//        startingPoint = 0.5 * Eigen::VectorXd::Ones(2);
//        N = Eigen::MatrixXd::Identity(2, 2);
//        shift = Eigen::VectorXd::Ones(2);
//        model = std::make_shared<ModelMock>();
//    }

    Eigen::MatrixXd A;
    Eigen::VectorXd b, startingPoint;
    std::shared_ptr<hops::Model> model;
};

BOOST_AUTO_TEST_SUITE(ProposalFactory)

    BOOST_AUTO_TEST_CASE(createUniformDikinWalk) {
        std::unique_ptr<hops::Proposal> markovChain;
        auto fixture = ProposalFactoryTestFixture();
        hops::ProposalFactory::createProposal<Eigen::MatrixXd,
//        BOOST_CHECK_NO_THROW(
        auto proposal = hops::ProposalFactory::createProposal<Eigen::MatrixXd, Eigen::VectorXd, hops::DikinProposal>(
                        fixture.A,
                        fixture.b,
                        fixture.startingPoint);
//        );
        BOOST_CHECK(proposal != nullptr);
    }


BOOST_AUTO_TEST_SUITE_END()
