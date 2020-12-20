#include <gtest/gtest.h>
#include <Eigen/Core>
#include <hops/Polytope/SimplexFactory.hpp>

namespace {
    TEST(Simpleactory, create2DSimplex) {
        Eigen::MatrixXd expectedA(3, 2);
        expectedA << -1, 0,
                0, -1,
                1, 1;
        Eigen::VectorXd expectedB(3);
        expectedB << 0, 0, 1;

        auto [A, b] = hops::SimplexFactory<Eigen::MatrixXd, Eigen::VectorXd>::createSimplex(2);

        EXPECT_EQ(A, expectedA);
        EXPECT_EQ(b, expectedB);
    }

    TEST(Simpleactory, create4DSimplex) {
        Eigen::MatrixXd expectedA(5, 4);
        expectedA << -1, 0, 0, 0,
                0, -1, 0, 0,
                0, 0, -1, 0,
                0, 0, 0, -1,
                1, 1, 1, 1;
        Eigen::VectorXd expectedB(5);
        expectedB << 0, 0, 0, 0, 1;

        auto [A, b] = hops::SimplexFactory<Eigen::MatrixXd, Eigen::VectorXd>::createSimplex(4);

        EXPECT_EQ(A, expectedA);
        EXPECT_EQ(b, expectedB);
    }
}
