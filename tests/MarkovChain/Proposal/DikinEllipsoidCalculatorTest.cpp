#include <gtest/gtest.h>
#include <Eigen/Core>
#include <hops/MarkovChain/Proposal/DikinEllipsoidCalculator.hpp>

namespace {
   TEST(DikinEllipsoidCalculator, Square) {
       const long rows = 6;
       const long cols = 3;
       Eigen::MatrixXd A(6, 3);
        A << 1, 0, 0,
                0, 1, 0,
                0, 0, 1,
                -1, 0, 0,
                0, -1, 0,
                0, 0, -1;
        Eigen::VectorXd b(rows);
        b << 1, 1, 1, 1, 1, 1;

        hops::DikinEllipsoidCalculator dikinEllipsoidCalculator(A, b);
        Eigen::VectorXd interiorPoint(cols);
        for (size_t i = 0; i < cols; ++i) {
            interiorPoint(i) = 0;
        }

        auto dikinEllipsoid = dikinEllipsoidCalculator.calculateDikinEllipsoid(interiorPoint);

        EXPECT_EQ(dikinEllipsoid, 2*Eigen::MatrixXd::Identity(cols, cols));
    }
}
