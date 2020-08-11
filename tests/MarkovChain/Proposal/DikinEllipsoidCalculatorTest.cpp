#include <gtest/gtest.h>
#include <Eigen/Core>
#include <hops/MarkovChain/Proposal/DikinEllipsoidCalculator.hpp>

namespace {
    TEST(DikinEllipsoidCalculator, Cube) {
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

        hops::DikinEllipsoidCalculator dikinEllipsoidCalculator(A, b);
        Eigen::VectorXd interiorPoint(cols);
        for (size_t i = 0; i < cols; ++i) {
            interiorPoint(i) = 0;
        }

        auto actualDikinEllipsoid = dikinEllipsoidCalculator.calculateDikinEllipsoid(interiorPoint);

        EXPECT_EQ(actualDikinEllipsoid, 2 * Eigen::MatrixXd::Identity(cols, cols));
    }

    TEST(DikinEllipsoidCalculator, Simplex) {
        const long rows = 4;
        const long cols = 3;

        Eigen::MatrixXd expectedDikinEllipsoid(cols, cols);
        expectedDikinEllipsoid << 3.35012345679, 2.56, 2.56,
                2.56, 3.35012345679, 2.56,
                2.56, 2.56, 3.35012345679;

        Eigen::MatrixXd A(rows, cols);
        A << 1, 1, 1,
                -1, 0, 0,
                0, -1, 0,
                0, 0, -1;
        Eigen::VectorXd b(rows);
        b << 1, 1, 1, 1;

        hops::DikinEllipsoidCalculator dikinEllipsoidCalculator(A, b);
        Eigen::VectorXd interiorPoint(cols);
        for (size_t i = 0; i < cols; ++i) {
            interiorPoint(i) = 1. / 8;
        }

        auto actualDikinEllipsoid = dikinEllipsoidCalculator.calculateDikinEllipsoid(interiorPoint);

        EXPECT_NEAR((actualDikinEllipsoid - expectedDikinEllipsoid).norm(), 0, 1e-10);
    }
}
