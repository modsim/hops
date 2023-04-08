#define BOOST_TEST_DYN_LINK
#define BOOST_TEST_MODULE dikinProposalTestSuite

#include <boost/test/unit_test.hpp>
#include <Eigen/Core>

#include "hops/MarkovChain/Proposal/DikinProposal.hpp"
#include "hops/RandomNumberGenerator/RandomNumberGenerator.hpp"

BOOST_AUTO_TEST_SUITE(DikinProposal)

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

        hops::DikinProposal dikinProposal(A, b, interiorPoint);

        std::vector<std::string> expectedNames = {"x_0", "x_1", "x_2"};
        auto actualNames = dikinProposal.getDimensionNames();

        BOOST_CHECK_EQUAL(actualNames.size(), expectedNames.size());
        for (size_t i = 0; i < expectedNames.size(); ++i) {
            BOOST_CHECK_EQUAL(actualNames[i], expectedNames[i]);
        }

        expectedNames = std::vector<std::string>{"y_1", "y_2", "y_3"};
        dikinProposal.setDimensionNames(expectedNames);
        actualNames = dikinProposal.getDimensionNames();

        BOOST_CHECK_EQUAL(actualNames.size(), expectedNames.size());
        for (size_t i = 0; i < expectedNames.size(); ++i) {
            BOOST_CHECK_EQUAL(actualNames[i], expectedNames[i]);
        }
    }

    BOOST_AUTO_TEST_CASE(Cube) {
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

        hops::DikinProposal dikinProposal(A, b, interiorPoint);
        hops::RandomNumberGenerator randomNumberGenerator((std::random_device()()));
        for (int i = 0; i < 100; ++i) {
            Eigen::VectorXd proposal = dikinProposal.propose(randomNumberGenerator);
            BOOST_CHECK(((b - A * proposal).array() > 0).all());
            if(std::isfinite(dikinProposal.computeLogAcceptanceProbability())) {
                dikinProposal.acceptProposal();
            }
        }
    }

BOOST_AUTO_TEST_SUITE_END()


