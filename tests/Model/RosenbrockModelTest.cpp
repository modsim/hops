#include <Eigen/Core>
#include <gtest/gtest.h>
#include <hops/Model/RosenbrockModel.hpp>

namespace {
    TEST(RosenbrockModel, calculateNegativeLogLikelihoodAtMinimum) {
        double expectedValue = 0;
        double scaleParameter = 1;
        Eigen::VectorXd shiftParameter = Eigen::VectorXd::Ones(2);

        hops::RosenbrockModel rosenbrockModel(scaleParameter, shiftParameter);
        Eigen::VectorXd evaluationPoint = Eigen::VectorXd::Ones(4);

        double actualValue = rosenbrockModel.calculateNegativeLogLikelihood(evaluationPoint);

        EXPECT_DOUBLE_EQ(actualValue, expectedValue);
    }

    TEST(RosenbrockModel, calculateNegativeLogLikelihoodAtMinimumWithScale) {
        double expectedValue = 0;
        double scaleParameter = 3.15;
        Eigen::VectorXd shiftParameter = Eigen::VectorXd::Ones(2);

        hops::RosenbrockModel rosenbrockModel(scaleParameter, shiftParameter);
        Eigen::VectorXd evaluationPoint = Eigen::VectorXd::Ones(4);

        double actualValue = rosenbrockModel.calculateNegativeLogLikelihood(evaluationPoint);

        EXPECT_DOUBLE_EQ(actualValue, expectedValue);
    }

    TEST(RosenbrockModel, calculateNegativeLogLikelihoodAtMinimumWithShift) {
        double expectedValue = 0;
        double scaleParameter = 1;
        Eigen::VectorXd shiftParameter(4);
        shiftParameter << -5, 1125, 3.14, 2.71;

        hops::RosenbrockModel rosenbrockModel(scaleParameter, shiftParameter);
        Eigen::VectorXd evaluationPoint(8);
        evaluationPoint << -5, 5 * 5, 1125, 1125 * 1125, 3.14, 3.14 * 3.14, 2.71, 2.71 * 2.71;

        double actualValue = rosenbrockModel.calculateNegativeLogLikelihood(evaluationPoint);

        EXPECT_NEAR(actualValue, expectedValue, 1e-10);
    }

    TEST(RosenbrockModel, calculateNegativeLogLikelihoodWithShift) {
        double expectedValue = 4483550.2037;
        double scaleParameter = 1;
        Eigen::VectorXd shiftParameter(4);
        shiftParameter << -5, 1125, 3.14, 2.71;

        hops::RosenbrockModel rosenbrockModel(scaleParameter, shiftParameter);
        Eigen::VectorXd evaluationPoint = 10 * Eigen::VectorXd::Ones(8);

        double actualValue = rosenbrockModel.calculateNegativeLogLikelihood(evaluationPoint);

        EXPECT_DOUBLE_EQ(actualValue, expectedValue);
    }

    TEST(RosenbrockModel, calculateNegativeLogLikelihoodWithShiftAndScale) {
        double expectedValue = 3.15 * 4483550.2037;
        double scaleParameter = 3.15;
        Eigen::VectorXd shiftParameter(4);
        shiftParameter << -5, 1125, 3.14, 2.71;

        hops::RosenbrockModel rosenbrockModel(scaleParameter, shiftParameter);
        Eigen::VectorXd evaluationPoint = 10 * Eigen::VectorXd::Ones(8);

        double actualValue = rosenbrockModel.calculateNegativeLogLikelihood(evaluationPoint);

        EXPECT_NEAR(actualValue, expectedValue, 1e-8);
    }

    TEST(RosenbrockModel, calculateLogLikelihoodGradientAtStationaryPoint) {
        Eigen::VectorXd expectedValue = Eigen::VectorXd::Zero(8);

        double scaleParameter = 1;
        Eigen::VectorXd shiftParameter(4);
        shiftParameter << -5, 1125, 3.14, 2.71;

        hops::RosenbrockModel rosenbrockModel(scaleParameter, shiftParameter);
        Eigen::VectorXd evaluationPoint(8);
        evaluationPoint << -5, 5 * 5, 1125, 1125 * 1125, 3.14, 3.14 * 3.14, 2.71, 2.71 * 2.71;

        Eigen::VectorXd actualValue = rosenbrockModel.calculateLogLikelihoodGradient(evaluationPoint);

        for (long i = 0; i < expectedValue.rows(); ++i) {
            EXPECT_NEAR(actualValue(i), expectedValue(i), 1e-10);
        }
    }

    TEST(RosenbrockModel, calculateLogLikelihoodGradient) {
        Eigen::VectorXd expectedValue(8);
        expectedValue << 720030, -18000, 717770, -18000, 720013.72, -18000, 720014.58, -18000;

        double scaleParameter = 1;
        Eigen::VectorXd shiftParameter(4);
        shiftParameter << -5, 1125, 3.14, 2.71;

        hops::RosenbrockModel rosenbrockModel(scaleParameter, shiftParameter);
        Eigen::VectorXd evaluationPoint = 10 * Eigen::VectorXd::Ones(8);

        Eigen::VectorXd actualValue = rosenbrockModel.calculateLogLikelihoodGradient(evaluationPoint);

        for (long i = 0; i < expectedValue.rows(); ++i) {
            EXPECT_DOUBLE_EQ(actualValue(i), expectedValue(i));
        }
    }

    TEST(RosenbrockModel, calculateLogLikelihoodGradientWithScale) {
        Eigen::VectorXd expectedValue(8);
        expectedValue << 720030, -18000, 717770, -18000, 720013.72, -18000, 720014.58, -18000;
        expectedValue = 3.105 * expectedValue;

        double scaleParameter = 3.105;
        Eigen::VectorXd shiftParameter(4);
        shiftParameter << -5, 1125, 3.14, 2.71;

        hops::RosenbrockModel rosenbrockModel(scaleParameter, shiftParameter);
        Eigen::VectorXd evaluationPoint = 10 * Eigen::VectorXd::Ones(8);

        Eigen::VectorXd actualValue = rosenbrockModel.calculateLogLikelihoodGradient(evaluationPoint);

        for (long i = 0; i < expectedValue.rows(); ++i) {
            EXPECT_DOUBLE_EQ(actualValue(i), expectedValue(i));
        }
    }

    TEST(RosenbrockModel, calculatHessianAtStationaryPoint) {
        Eigen::MatrixXd expectedValue = Eigen::MatrixXd::Zero(6, 6);
        expectedValue(0, 0) = 1200 * 5 * 5 - 400 * 5 * 5 + 2;
        expectedValue(1, 0) = -400 * -5;
        expectedValue(0, 1) = -400 * -5;
        expectedValue(1, 1) = 200;

        expectedValue(2, 2) = 1200 * 1125 * 1125 - 400 * 1125 * 1125 + 2;
        expectedValue(3, 2) = -400 * 1125;
        expectedValue(2, 3) = -400 * 1125;
        expectedValue(3, 3) = 200;

        expectedValue(4, 4) = 1200 * 3.14 * 3.14 - 400 * 3.14 * 3.14 + 2;
        expectedValue(5, 4) = -400 * 3.14;
        expectedValue(4, 5) = -400 * 3.14;
        expectedValue(5, 5) = 200;

        double scaleParameter = 1;
        Eigen::VectorXd shiftParameter(3);
        shiftParameter << -5, 1125, 3.14;

        hops::RosenbrockModel rosenbrockModel(scaleParameter, shiftParameter);
        Eigen::VectorXd evaluationPoint(6);
        evaluationPoint << -5, 5 * 5, 1125, 1125 * 1125, 3.14, 3.14 * 3.14;

        Eigen::MatrixXd actualValue = rosenbrockModel.calculateHessian(evaluationPoint);

        for (long i = 0; i < std::max(actualValue.rows(), expectedValue.rows()); ++i) {
            for (long j = 0; j < std::max(actualValue.cols(), expectedValue.cols()); ++j) {
                if(std::abs(actualValue(i, j) - expectedValue(i, j)) > 1) {
                   std::cout << i << ", " << j << std::endl;
                }
                EXPECT_DOUBLE_EQ(actualValue(i, j), expectedValue(i, j));
            }
        }
    }

    TEST(RosenbrockModel, calculateHessianAtStationaryWithScale) {
        Eigen::MatrixXd expectedValue = Eigen::MatrixXd::Zero(6, 6);
        expectedValue(0, 0) = 1200 * 5 * 5 - 400 * 5 * 5 + 2;
        expectedValue(1, 0) = -400 * -5;
        expectedValue(0, 1) = -400 * -5;
        expectedValue(1, 1) = 200;

        expectedValue(2, 2) = 1200 * 1125 * 1125 - 400 * 1125 * 1125 + 2;
        expectedValue(3, 2) = -400 * 1125;
        expectedValue(2, 3) = -400 * 1125;
        expectedValue(3, 3) = 200;

        expectedValue(4, 4) = 1200 * 3.14 * 3.14 - 400 * 3.14 * 3.14 + 2;
        expectedValue(5, 4) = -400 * 3.14;
        expectedValue(4, 5) = -400 * 3.14;
        expectedValue(5, 5) = 200;

        double scaleParameter = 3.51;
        expectedValue = scaleParameter * expectedValue;
        Eigen::VectorXd shiftParameter(3);
        shiftParameter << -5, 1125, 3.14;

        hops::RosenbrockModel rosenbrockModel(scaleParameter, shiftParameter);
        Eigen::VectorXd evaluationPoint(6);
        evaluationPoint << -5, 5 * 5, 1125, 1125 * 1125, 3.14, 3.14 * 3.14;

        Eigen::MatrixXd actualValue = rosenbrockModel.calculateHessian(evaluationPoint);

        for (long i = 0; i < std::max(actualValue.rows(), expectedValue.rows()); ++i) {
            for (long j = 0; j < std::max(actualValue.cols(), expectedValue.cols()); ++j) {
                if(std::abs(actualValue(i, j) - expectedValue(i, j)) > 1) {
                    std::cout << i << ", " << j << std::endl;
                }
                EXPECT_DOUBLE_EQ(actualValue(i, j), expectedValue(i, j));
            }
        }
    }
}
