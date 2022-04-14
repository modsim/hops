#define BOOST_TEST_MODULE CSmMALAProposalTestSuite
#define BOOST_TEST_DYN_LINK

#include <boost/test/included/unit_test.hpp>
#include <numeric>
#include <Eigen/Core>

#include <hops/MarkovChain/Proposal/CSmMALAProposal.hpp>
#include <hops/MarkovChain/MarkovChainAdapter.hpp>
#include <hops/MarkovChain/Draw/MetropolisHastingsFilter.hpp>
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
            if (((b - A * proposal).array() > 0).all()) {
                proposer.acceptProposal();
            }
        }

        Eigen::VectorXd proposal = proposer.propose(randomNumberGenerator);
        Eigen::VectorXd expectedProposal(3);
        expectedProposal << 0.9999741429740745, -0.98150954639435795, -0.99999996305021499;
        BOOST_CHECK(proposal.isApprox(expectedProposal));

        BOOST_CHECK(proposer.getModel() != nullptr);
    }


    BOOST_AUTO_TEST_CASE(GaussianInWideCubeHasCorrectStd) {
        const long rows = 2;
        const long cols = 1;
        Eigen::MatrixXd A(rows, cols);
        A << 1, -1;
        Eigen::VectorXd b = 1000 * Eigen::VectorXd::Ones(rows);
        Eigen::VectorXd interiorPoint(cols);
        for (size_t i = 0; i < cols; ++i) {
            interiorPoint(i) = 0;
        }

        auto model = hops::Gaussian(interiorPoint, Eigen::MatrixXd::Identity(cols, cols));

        auto mc = hops::MarkovChainAdapter(
                hops::MetropolisHastingsFilter(
                        hops::CSmMALAProposal(A,
                                              b,
                                              interiorPoint,
                                              model)
                )
        );

        hops::RandomNumberGenerator randomNumberGenerator(42);
        std::vector<double> samples;
        double num_samples = 50'000'000;
        for (int i = 0; i < num_samples; ++i) {
            auto [alpha, state] = mc.draw(randomNumberGenerator);
            samples.emplace_back(state(0));
        }

        double mean = std::accumulate(samples.begin(), samples.end(), 0.) / samples.size();

        double sq_sum = 0;
        for (const auto &s:samples) {
            sq_sum += (s - mean) * (s - mean);
        }
        double stdev = std::sqrt(sq_sum / (samples.size() - 1));

        std::cout << mean << std::endl;
        BOOST_CHECK_CLOSE(stdev, 1., 1);
    }

BOOST_AUTO_TEST_SUITE_END()

