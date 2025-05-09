#define BOOST_TEST_DYN_LINK
#define BOOST_TEST_MODULE DikinEllipsoidCalculatorTestSuite

#include <boost/test/unit_test.hpp>
#include <Eigen/Core>

#include "hops/MarkovChain/Proposal/DikinEllipsoidCalculator.hpp"

BOOST_AUTO_TEST_SUITE(DikinEllipsoidCalculator)

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

        hops::DikinEllipsoidCalculator dikinEllipsoidCalculator(A, b);
        Eigen::VectorXd interiorPoint(cols);
        for (size_t i = 0; i < cols; ++i) {
            interiorPoint(i) = 0;
        }

        auto actualDikinEllipsoid = dikinEllipsoidCalculator.computeDikinEllipsoid(interiorPoint);

        BOOST_CHECK(actualDikinEllipsoid == 2 * Eigen::MatrixXd::Identity(cols, cols));
    }

    BOOST_AUTO_TEST_CASE(CholeskyOfCube) {
        const long rows = 6;
        const long cols = 3;
        Eigen::MatrixXd expectedDikinEllipsoid = 2 * Eigen::MatrixXd::Identity(cols, cols);

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

        auto[choleskyWasSuccessful, actualDikinEllipsoidLowerFactor] = dikinEllipsoidCalculator.computeCholeskyFactorOfDikinEllipsoid(
                interiorPoint);
        Eigen::MatrixXd actualDikinEllipsoid =
                actualDikinEllipsoidLowerFactor * actualDikinEllipsoidLowerFactor.transpose();

        BOOST_CHECK(choleskyWasSuccessful);
        BOOST_CHECK(((actualDikinEllipsoid - expectedDikinEllipsoid).array() < 1e-15).all());
    }

    BOOST_AUTO_TEST_CASE(Simplex) {
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

        auto actualDikinEllipsoid = dikinEllipsoidCalculator.computeDikinEllipsoid(interiorPoint);

        BOOST_CHECK(((actualDikinEllipsoid - expectedDikinEllipsoid).array() < 1e-12).all());
    }

    BOOST_AUTO_TEST_CASE(CholeskyOfSimplex) {
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

        auto[choleskyWasSuccessful, actualDikinEllipsoidLowerFactor] = dikinEllipsoidCalculator.computeCholeskyFactorOfDikinEllipsoid(
                interiorPoint);
        Eigen::MatrixXd actualDikinEllipsoid =
                actualDikinEllipsoidLowerFactor * actualDikinEllipsoidLowerFactor.transpose();

        BOOST_CHECK(choleskyWasSuccessful);
        BOOST_CHECK(((actualDikinEllipsoid - expectedDikinEllipsoid).array() < 1e-12).all());
    }

BOOST_AUTO_TEST_SUITE_END()
