#define BOOST_TEST_MODULE CSmMALAProposalTestSuite
#define BOOST_TEST_DYN_LINK

#include <boost/test/included/unit_test.hpp>
#include <Eigen/Core>

#include <hops/MarkovChain/Proposal/CSmMALAProposal.hpp>
#include <hops/Model/Gaussian.hpp>
#include <hops/RandomNumberGenerator/RandomNumberGenerator.hpp>

BOOST_AUTO_TEST_SUITE(CSmMALAProposal)

    BOOST_AUTO_TEST_CASE(GaussianInCube) {
        const long rows = 6;
        const long cols = 3;
        Eigen::MatrixXd A(rows, cols);
        A << 1, 0, 0,
                0, 1, 0,
                0, 0, 1,
                -1, 0, 0,
                0, -1, 0,
                0, 0, -1;
        Eigen::VectorXd b(rows);
        b << 1, 1, 1, 1, 1, 1;
        Eigen::VectorXd interiorPoint(cols);
        for (size_t i = 0; i < cols; ++i) {
            interiorPoint(i) = 0;
        }

        auto model = hops::Gaussian(interiorPoint, Eigen::MatrixXd::Identity(cols, cols));

        hops::CSmMALAProposal proposer(A,
                                       b,
                                       interiorPoint,
                                       model);

        hops::RandomNumberGenerator randomNumberGenerator(42);
        for (int i = 0; i < 100; ++i) {
            Eigen::VectorXd proposal = proposer.propose(randomNumberGenerator);
            double acceptanceChance = proposer.computeLogAcceptanceProbability();
            BOOST_CHECK(std::exp(acceptanceChance) >= 0);
            if(((b - A * proposal).array() > 0).all()) {
                proposer.acceptProposal();
            }
        }

        Eigen::VectorXd proposal = proposer.propose(randomNumberGenerator);
        Eigen::VectorXd expectedProposal(3);
        std::cout << std::setprecision(17) << proposal.transpose() << std::endl;
        std::cout << std::setprecision(17) << expectedProposal.transpose() << std::endl;
        std::cout << std::setprecision(17) << (proposal-expectedProposal).transpose() << std::endl;
        BOOST_CHECK(proposal.isApprox(expectedProposal));

        BOOST_CHECK(proposer.getModel() != nullptr);
    }

BOOST_AUTO_TEST_SUITE_END()

