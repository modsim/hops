#define BOOST_TEST_DYN_LINK
#define BOOST_TEST_MODULE RosenbrockTestSuite

#include <boost/test/unit_test.hpp>
#include <Eigen/Core>

#include "hops/Model/Rosenbrock.hpp"

BOOST_AUTO_TEST_SUITE(RosenbrockTestSuite)

    BOOST_AUTO_TEST_CASE(CalculateNegativeLogLikelihoodAtMinimum) {
        double expectedValue = 0;
        double scaleParameter = 1;
        hops::VectorType shiftParameter = hops::VectorType::Ones(2);

        hops::Rosenbrock rosenbrock(scaleParameter, shiftParameter);
        hops::VectorType evaluationPoint = hops::VectorType::Ones(4);

        double actualValue = rosenbrock.computeNegativeLogLikelihood(evaluationPoint);

        BOOST_CHECK_EQUAL(actualValue, expectedValue);
    }

    BOOST_AUTO_TEST_CASE(CalculateNegativeLogLikelihoodAtMinimumWithScale) {
        double expectedValue = 0;
        double scaleParameter = 3.15;
        hops::VectorType shiftParameter = hops::VectorType::Ones(2);

        hops::Rosenbrock rosenbrock(scaleParameter, shiftParameter);
        hops::VectorType evaluationPoint = hops::VectorType::Ones(4);

        double actualValue = rosenbrock.computeNegativeLogLikelihood(evaluationPoint);

        BOOST_CHECK_EQUAL(actualValue, expectedValue);
    }

    BOOST_AUTO_TEST_CASE(CalculateNegativeLogLikelihoodAtMinimumWithShift) {
        double expectedValue = 0;
        double scaleParameter = 1;
        hops::VectorType shiftParameter(4);
        shiftParameter << -5, 1125, 3.14, 2.71;

        hops::Rosenbrock rosenbrock(scaleParameter, shiftParameter);
        hops::VectorType evaluationPoint(8);
        evaluationPoint << -5, 5 * 5, 1125, 1125 * 1125, 3.14, 3.14 * 3.14, 2.71, 2.71 * 2.71;

        double actualValue = rosenbrock.computeNegativeLogLikelihood(evaluationPoint);

        BOOST_CHECK_SMALL(actualValue - expectedValue, 1e-8);
    }

    BOOST_AUTO_TEST_CASE(CalculateNegativeLogLikelihoodWithShift) {
        double expectedValue = 4483550.2037;
        double scaleParameter = 1;
        hops::VectorType shiftParameter(4);
        shiftParameter << -5, 1125, 3.14, 2.71;

        hops::Rosenbrock rosenbrock(scaleParameter, shiftParameter);
        hops::VectorType evaluationPoint = 10 * hops::VectorType::Ones(8);

        double actualValue = rosenbrock.computeNegativeLogLikelihood(evaluationPoint);

        BOOST_CHECK_EQUAL(actualValue, expectedValue);
    }

    BOOST_AUTO_TEST_CASE(CalculateNegativeLogLikelihoodWithShiftAndScale) {
        double expectedValue = 3.15 * 4483550.2037;
        double scaleParameter = 3.15;
        hops::VectorType shiftParameter(4);
        shiftParameter << -5, 1125, 3.14, 2.71;

        hops::Rosenbrock rosenbrock(scaleParameter, shiftParameter);
        hops::VectorType evaluationPoint = 10 * hops::VectorType::Ones(8);

        double actualValue = rosenbrock.computeNegativeLogLikelihood(evaluationPoint);

        BOOST_CHECK_SMALL(actualValue - expectedValue, 1e-8);
    }

    BOOST_AUTO_TEST_CASE(CalculateLogLikelihoodGradientAtStationaryPoint) {
        hops::VectorType expectedValue = hops::VectorType::Zero(8);

        double scaleParameter = 1;
        hops::VectorType shiftParameter(4);
        shiftParameter << -5, 1125, 3.14, 2.71;

        hops::Rosenbrock rosenbrock(scaleParameter, shiftParameter);
        hops::VectorType evaluationPoint(8);
        evaluationPoint << -5, 5 * 5, 1125, 1125 * 1125, 3.14, 3.14 * 3.14, 2.71, 2.71 * 2.71;

        auto actualValue = rosenbrock.computeLogLikelihoodGradient(evaluationPoint);
        if(actualValue) {
            BOOST_CHECK(actualValue.value().isApprox(expectedValue));
        } else {
            BOOST_FAIL("Gradient optional should not be empty.");
        }
    }

    BOOST_AUTO_TEST_CASE(CalculateLogLikelihoodGradient) {
        hops::VectorType expectedValue(8);
        expectedValue << 720030, -18000, 717770, -18000, 720013.72, -18000, 720014.58, -18000;

        double scaleParameter = 1;
        hops::VectorType shiftParameter(4);
        shiftParameter << -5, 1125, 3.14, 2.71;

        hops::Rosenbrock rosenbrock(scaleParameter, shiftParameter);
        hops::VectorType evaluationPoint = 10 * hops::VectorType::Ones(8);

        auto actualValue = rosenbrock.computeLogLikelihoodGradient(evaluationPoint);
        if(actualValue) {
            BOOST_CHECK(actualValue.value().isApprox(expectedValue));
        } else {
            BOOST_FAIL("Gradient optional should not be empty.");
        }
    }

    BOOST_AUTO_TEST_CASE(CalculateLogLikelihoodGradientWithScale) {
        hops::VectorType expectedValue(8);
        expectedValue << 720030, -18000, 717770, -18000, 720013.72, -18000, 720014.58, -18000;
        expectedValue = 3.105 * expectedValue;

        double scaleParameter = 3.105;
        hops::VectorType shiftParameter(4);
        shiftParameter << -5, 1125, 3.14, 2.71;

        hops::Rosenbrock rosenbrock(scaleParameter, shiftParameter);
        hops::VectorType evaluationPoint = 10 * hops::VectorType::Ones(8);

        auto actualValue = rosenbrock.computeLogLikelihoodGradient(evaluationPoint); if(actualValue) {
            BOOST_CHECK(actualValue.value().isApprox(expectedValue));
        } else {
            BOOST_FAIL("Gradient optional should not be empty.");
        }
    }

    BOOST_AUTO_TEST_CASE(CalculatHessianAtStationaryPoint) {
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
        hops::VectorType shiftParameter(3);
        shiftParameter << -5, 1125, 3.14;

        hops::Rosenbrock rosenbrock(scaleParameter, shiftParameter);
        hops::VectorType evaluationPoint(6);
        evaluationPoint << -5, 5 * 5, 1125, 1125 * 1125, 3.14, 3.14 * 3.14;

        Eigen::MatrixXd actualValue = rosenbrock.computeHessian(evaluationPoint);

        BOOST_CHECK(actualValue.isApprox(expectedValue));
    }

    BOOST_AUTO_TEST_CASE(CalculateHessianAtStationaryWithScale) {
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
        hops::VectorType shiftParameter(3);
        shiftParameter << -5, 1125, 3.14;

        hops::Rosenbrock rosenbrock(scaleParameter, shiftParameter);
        hops::VectorType evaluationPoint(6);
        evaluationPoint << -5, 5 * 5, 1125, 1125 * 1125, 3.14, 3.14 * 3.14;

        Eigen::MatrixXd actualValue = rosenbrock.computeHessian(evaluationPoint);

        BOOST_CHECK(actualValue.isApprox(expectedValue));
    }

BOOST_AUTO_TEST_SUITE_END()
