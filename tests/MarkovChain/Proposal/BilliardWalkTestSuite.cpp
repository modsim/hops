#define BOOST_TEST_DYN_LINK
#define BOOST_TEST_MODULE BilliardWalkProposalTestSuite

#include <boost/test/unit_test.hpp>
#include <chrono>
#include <Eigen/Core>

#include "hops/MarkovChain/Draw/MetropolisHastingsFilter.hpp"
#include "hops/MarkovChain/MarkovChainAdapter.hpp"
#include "hops/MarkovChain/Proposal/BilliardWalkProposal.hpp"
#include "hops/RandomNumberGenerator/RandomNumberGenerator.hpp"

BOOST_AUTO_TEST_SUITE(BilliardWalkProposal)

    BOOST_AUTO_TEST_CASE(DimensionNames) {
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

        long max_reflections = 10;
        hops::BilliardWalkProposal proposer(A,
                                            b,
                                            interiorPoint,
                                            max_reflections);

        std::vector<std::string> expectedNames = {"x_0", "x_1", "x_2"};
        auto actualNames = proposer.getDimensionNames();

        BOOST_CHECK_EQUAL(actualNames.size(), expectedNames.size());
        for(size_t i=0; i<expectedNames.size(); ++i) {
            BOOST_CHECK_EQUAL(actualNames[i], expectedNames[i]);
        }

        expectedNames = std::vector<std::string>{"y_1", "y_2", "y_3"};
        proposer.setDimensionNames(expectedNames);
        actualNames = proposer.getDimensionNames();

        BOOST_CHECK_EQUAL(actualNames.size(), expectedNames.size());
        for(size_t i=0; i<expectedNames.size(); ++i) {
            BOOST_CHECK_EQUAL(actualNames[i], expectedNames[i]);
        }
    }

    BOOST_AUTO_TEST_CASE(Cube) {
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

        long max_reflections = 10;
        hops::BilliardWalkProposal proposer(A,
                                            b,
                                            interiorPoint,
                                            max_reflections);


        hops::RandomNumberGenerator randomNumberGenerator(42);
        auto t1 = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < 100; ++i) {
            Eigen::VectorXd proposal = proposer.propose(randomNumberGenerator);
            double acceptanceChance = proposer.computeLogAcceptanceProbability();
            BOOST_CHECK(((b - A * proposal).array() > 0).all());
            BOOST_CHECK(std::exp(acceptanceChance) >= 0);
            proposer.acceptProposal();
        }
        auto t2 = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double, std::milli> ms_double = t2 - t1;
    }


BOOST_AUTO_TEST_SUITE_END()
