#define BOOST_TEST_MODULE NormalizePolytopeTestSuite
#define BOOST_TEST_DYN_LINK

#include <boost/test/included/unit_test.hpp>
#include <cmath>
#include <hops/Polytope/NormalizePolytope.hpp>

BOOST_AUTO_TEST_SUITE(NormalizePolytope)

    BOOST_AUTO_TEST_CASE(NormalizeSimplex) {
        Eigen::MatrixXd expectedA(4, 3);
        expectedA << 1. / std::sqrt(3), 1. / std::sqrt(3), 1. / std::sqrt(3),
                1, 0, 0,
                0, 1, 0,
                0, 0, 1;
        Eigen::VectorXd expectedB(4);
        expectedB << 1. / std::sqrt(3), 1, 1, 1;

        Eigen::MatrixXd A(4, 3);
        A << 1, 1, 1,
                1, 0, 0,
                0, 1, 0,
                0, 0, 1;
        Eigen::VectorXd b = Eigen::VectorXd::Ones(4);

        hops::normalizePolytope(A, b);

        BOOST_CHECK(A.isApprox(expectedA));
        BOOST_CHECK(b.isApprox(expectedB));
    }

BOOST_AUTO_TEST_SUITE_END()
