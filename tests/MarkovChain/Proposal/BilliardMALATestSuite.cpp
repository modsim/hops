#define BOOST_TEST_MODULE BilliardMALAProposalTestSuite
#define BOOST_TEST_DYN_LINK

#include <boost/test/included/unit_test.hpp>
#include <Eigen/Core>

#include <hops/MarkovChain/Proposal/BilliardMALAProposal.hpp>
#include <hops/Model/Rosenbrock.hpp>
#include <hops/RandomNumberGenerator/RandomNumberGenerator.hpp>

BOOST_AUTO_TEST_SUITE(BilliardMALAProposal)

    BOOST_AUTO_TEST_CASE(RosenBrockInCube) {
        const long rows = 8;
        const long cols = 4;
        Eigen::MatrixXd A(rows, cols);
        A << 1, 0, 0, 0,
                0, 1, 0, 0,
                0, 0, 1, 0,
                0, 0, 0, 1,
                -1, 0, 0, 0,
                0, -1, 0, 0,
                0, 0, -1, 0,
                0, 0, 0, -1;
        Eigen::VectorXd b(rows);
        b << 1, 1, 1, 1, 1, 1, 1, 1;
        Eigen::VectorXd interiorPoint(cols);
        for (size_t i = 0; i < cols; ++i) {
            interiorPoint(i) = 0;
        }

        auto model = hops::Rosenbrock(1, Eigen::VectorXd::Zero(cols / 2));

        long max_reflections = 10;
        hops::BilliardMALAProposal proposer(A,
                                            b,
                                            interiorPoint,
                                            model,
                                            max_reflections);

        hops::RandomNumberGenerator randomNumberGenerator((std::random_device()()));
        for (int i = 0; i < 100; ++i) {
            Eigen::VectorXd proposal = proposer.propose(randomNumberGenerator);
            double acceptanceChance = proposer.computeLogAcceptanceProbability();
            BOOST_CHECK(((b - A * proposal).array() > 0).all());
            BOOST_CHECK(std::exp(acceptanceChance) >= 0);
            proposer.acceptProposal();
        }

        BOOST_CHECK(proposer.getModel() != nullptr);
    }

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


        long max_reflections = 10;
        hops::BilliardMALAProposal proposer(A,
                                            b,
                                            interiorPoint,
                                            model,
                                            max_reflections);

        hops::RandomNumberGenerator randomNumberGenerator((std::random_device()()));
        for (int i = 0; i < 100; ++i) {
            Eigen::VectorXd proposal = proposer.propose(randomNumberGenerator);
            double acceptanceChance = proposer.computeLogAcceptanceProbability();
            BOOST_CHECK(((b - A * proposal).array() > 0).all());
            BOOST_CHECK(std::exp(acceptanceChance) >= 0);
            proposer.acceptProposal();
        }

        BOOST_CHECK(proposer.getModel() != nullptr);
    }

    BOOST_AUTO_TEST_CASE(ColdnessGaussianInCube) {
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

        auto model = hops::Coldness(hops::Gaussian(interiorPoint, Eigen::MatrixXd::Identity(cols, cols)), 0.5);


        long max_reflections = 10;
        hops::BilliardMALAProposal proposer(A,
                                            b,
                                            interiorPoint,
                                            model,
                                            max_reflections);

        hops::RandomNumberGenerator randomNumberGenerator((std::random_device()()));
        for (int i = 0; i < 100; ++i) {
            Eigen::VectorXd proposal = proposer.propose(randomNumberGenerator);
            double acceptanceChance = proposer.computeLogAcceptanceProbability();
            BOOST_CHECK(((b - A * proposal).array() > 0).all());
            BOOST_CHECK(std::exp(acceptanceChance) >= 0);
            proposer.acceptProposal();
        }

        BOOST_CHECK(proposer.getModel() != nullptr);
    }


BOOST_AUTO_TEST_SUITE_END()

