#include <gtest/gtest.h>
#include <cmath>
#include <hops/Polytope/NormalizePolytope.hpp>

namespace {
    TEST(NormalizePolytope, normalizeSimplex) {
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

        EXPECT_TRUE(A.isApprox(expectedA));
        EXPECT_TRUE(b.isApprox(expectedB));
    }
}
