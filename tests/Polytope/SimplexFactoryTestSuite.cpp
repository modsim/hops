#define BOOST_TEST_DYN_LINK
#define BOOST_TEST_MODULE SimplexFactoryTestSuite

#include <boost/test/unit_test.hpp>
#include <Eigen/Core>

#include "hops/Polytope/SimplexFactory.hpp"


BOOST_AUTO_TEST_SUITE(SimplexFactoryTestSuite)

    BOOST_AUTO_TEST_CASE(Create2DSimplex) {
        Eigen::MatrixXd expectedA(3, 2);
        expectedA << -1, 0,
                0, -1,
                1, 1;
        Eigen::VectorXd expectedB(3);
        expectedB << 0, 0, 1;

        auto[A, b] = hops::SimplexFactory<Eigen::MatrixXd, Eigen::VectorXd>::createSimplex(2);

        BOOST_CHECK_EQUAL(A, expectedA);
        BOOST_CHECK_EQUAL(b, expectedB);
    }

    BOOST_AUTO_TEST_CASE(Create4DSimplex) {
        Eigen::MatrixXd expectedA(5, 4);
        expectedA << -1, 0, 0, 0,
                0, -1, 0, 0,
                0, 0, -1, 0,
                0, 0, 0, -1,
                1, 1, 1, 1;
        Eigen::VectorXd expectedB(5);
        expectedB << 0, 0, 0, 0, 1;

        auto[A, b] = hops::SimplexFactory<Eigen::MatrixXd, Eigen::VectorXd>::createSimplex(4);

        BOOST_CHECK_EQUAL(A, expectedA);
        BOOST_CHECK_EQUAL(b, expectedB);
    }

BOOST_AUTO_TEST_SUITE_END()
