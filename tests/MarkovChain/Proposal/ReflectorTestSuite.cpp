#define BOOST_TEST_DYN_LINK
#define BOOST_TEST_MODULE ReflectorTestSuite

#include <boost/test/unit_test.hpp>

#include "hops/MarkovChain/Proposal/Reflector.hpp"
#include "hops/Utility/VectorType.hpp"

BOOST_AUTO_TEST_SUITE(Reflector)

    BOOST_AUTO_TEST_CASE(TestNoReflectionRequired) {
        Eigen::MatrixXd A(1, 1);
        A << -1;
        hops::VectorType b(1);
        b << 0;

        hops::VectorType startPoint(1);
        startPoint << 1;
        hops::VectorType endPoint(1);
        endPoint << 2;

        auto[reflectionSuccessful, numReflections, reflectedPoint] =
        hops::Reflector::reflectIntoPolytope(A, b, startPoint, endPoint, 200);

        BOOST_CHECK(reflectionSuccessful);
        BOOST_CHECK_EQUAL(numReflections, 0);
        BOOST_CHECK_EQUAL(reflectedPoint, endPoint);

    }

    BOOST_AUTO_TEST_CASE(Test1ReflectionStep) {
        // Constrain to positive part of real numbers, representation is Ax<b
        Eigen::MatrixXd A(1, 1);
        A << -1;
        hops::VectorType b(1);
        b << 0;

        hops::VectorType startPoint(1);
        startPoint << 1;
        hops::VectorType endPoint(1);
        endPoint << -9; // endpoint outside of constraints

        auto[reflectionSuccessful, numReflections, reflectedPoint] =
        hops::Reflector::reflectIntoPolytope(A, b, startPoint, endPoint, 200);

        BOOST_CHECK(reflectionSuccessful);
        BOOST_CHECK_EQUAL(numReflections, 1);
        // In this example it happens that the reflectPoint is equal to startPoint due to symmetry
        hops::VectorType expectedResult(1);
        expectedResult << 9;

        BOOST_CHECK(reflectedPoint.isApprox(expectedResult));
    }

    BOOST_AUTO_TEST_CASE(Test1ReflectionStepWithBigEpsilon) {
        // Constrain to positive part of real numbers, representation is Ax<b
        Eigen::MatrixXd A(1, 1);
        A << -1;
        hops::VectorType b(1);
        b << 0;

        hops::VectorType startPoint(1);
        startPoint << 1;
        hops::VectorType endPoint(1);
        endPoint << -9; // endpoint outside of constraints

        auto[reflectionSuccessful, numReflections, reflectedPoint] =
        hops::Reflector::reflectIntoPolytope(A, b, startPoint, endPoint, 200);

        BOOST_CHECK(reflectionSuccessful);
        BOOST_CHECK_EQUAL(numReflections, 1);
        // In this example it happens that the reflectPoint is equal to startPoint due to symmetry
        hops::VectorType expectedResult(1);
        expectedResult << 9;

        BOOST_CHECK(reflectedPoint.isApprox(expectedResult));
    }

    BOOST_AUTO_TEST_CASE(TestSeveralReflectionsInSimplexWhenMaxLimitIsHit) {
        // Constrain to positive part of real numbers, representation is Ax<b
        Eigen::MatrixXd A(3, 2);
        A << -1, 0, 0, -1, 1, 1;
        hops::VectorType b(3);
        b << 0, 0, 1;

        hops::VectorType startPoint(2);
        startPoint << 0.25, 0.25;
        hops::VectorType endPoint(2);
        endPoint << 1000, 1000; // endpoint outside of constraints

        auto[reflectionSuccessful, numReflections, reflectedPoint] =
        hops::Reflector::reflectIntoPolytope(A, b, startPoint, endPoint, 20);

        BOOST_CHECK_EQUAL(reflectionSuccessful, false);
        BOOST_CHECK_EQUAL(numReflections, 20);
        // In this example it happens that the reflectPoint is equal to startPoint due to symmetry
        hops::VectorType expectedResult = reflectedPoint;

        BOOST_CHECK_EQUAL(reflectedPoint, expectedResult);
    }

    BOOST_AUTO_TEST_CASE(TestTwoReflectionsInSimplex) {
        // Constrain to positive part of real numbers, representation is Ax<b
        Eigen::MatrixXd A(3, 2);
        A << -1, 0, 0, -1, 1, 1;
        hops::VectorType b(3);
        b << 0, 0, 1;

        hops::VectorType startPoint(2);
        startPoint << 0.25, 0.25;
        hops::VectorType endPoint(2);
        endPoint << 1, 1; // endpoint outside of constraints

        long expectedNumReflections = std::floor(1. / 0.5);

        auto[reflectionSuccessful, actualNumReflections, reflectedPoint] =
        hops::Reflector::reflectIntoPolytope(A, b, startPoint, endPoint, expectedNumReflections + 1);

        BOOST_CHECK(reflectionSuccessful);
        BOOST_CHECK_EQUAL(actualNumReflections, expectedNumReflections);

        // expectedResult is at 0 because first reflection happens at 0.5, 0.5. Then the distsance between
        // 0,0 and 0.5, 0.5 is travelled exactly 19 times.
        hops::VectorType expectedResult = Eigen::VectorXd::Zero(endPoint.rows());

        BOOST_CHECK(reflectedPoint.isApprox(expectedResult));
    }

    BOOST_AUTO_TEST_CASE(TestSeveralReflectionsInSimplex) {
        // Constrain to positive part of real numbers, representation is Ax<b
        Eigen::MatrixXd A(3, 2);
        A << -1, 0, 0, -1, 1, 1;
        hops::VectorType b(3);
        b << 0, 0, 1;

        hops::VectorType startPoint(2);
        startPoint << 0.25, 0.25;
        hops::VectorType endPoint(2);
        endPoint << 10, 10; // endpoint outside of constraints

        long expectedNumReflections = std::floor((10) / 0.5);

        auto[reflectionSuccessful, actualNumReflections, reflectedPoint] =
        hops::Reflector::reflectIntoPolytope(A, b, startPoint, endPoint, expectedNumReflections * 5);

        BOOST_CHECK(reflectionSuccessful);
        BOOST_CHECK_EQUAL(actualNumReflections, expectedNumReflections);

        // expectedResult is at 0 because first reflection happens at 0.5, 0.5. After this first reflection,
        // the distance between 0,0 and 0.5, 0.5 is travelled an additional 19 times and it stops at the origin.
        hops::VectorType expectedResult = Eigen::VectorXd::Zero(endPoint.rows());

        BOOST_CHECK(reflectedPoint.isApprox(expectedResult));
    }

    BOOST_AUTO_TEST_CASE(TestHundredsOfReflectionsInSimplex) {
        // Constrain to positive part of real numbers, representation is Ax<b
        Eigen::MatrixXd A(3, 2);
        A << -1, 0, 0, -1, 1, 1;
        hops::VectorType b(3);
        b << 0, 0, 1;

        hops::VectorType startPoint(2);
        startPoint << 0.25, 0.25;
        hops::VectorType endPoint(2);
        endPoint << 100, 100; // endpoint outside of constraints

        long expectedNumReflections = std::floor((100.) / 0.5);

        auto[reflectionSuccessful, actualNumReflections, reflectedPoint] =
        hops::Reflector::reflectIntoPolytope(A, b, startPoint, endPoint, 2 * expectedNumReflections);

        BOOST_CHECK(reflectionSuccessful);
        BOOST_CHECK_EQUAL(actualNumReflections, expectedNumReflections);

        hops::VectorType expectedResult = Eigen::VectorXd::Zero(endPoint.rows());

        BOOST_CHECK(reflectedPoint.isApprox(expectedResult));
    }

    BOOST_AUTO_TEST_CASE(TestThousandsOfReflectionsInSimplex) {
        // Constrain to positive part of real numbers, representation is Ax<b
        Eigen::MatrixXd A(3, 2);
        A << -1, 0, 0, -1, 1, 1;
        hops::VectorType b(3);
        b << 0, 0, 1;

        hops::VectorType startPoint(2);
        startPoint << 0.25, 0.25;
        hops::VectorType endPoint(2);
        endPoint << 1000, 1000; // endpoint outside of constraints

        long expectedNumReflections = std::floor((1000.) / 0.5);

        auto[reflectionSuccessful, actualNumReflections, reflectedPoint] =
        hops::Reflector::reflectIntoPolytope(A, b, startPoint, endPoint, expectedNumReflections + 1);

        BOOST_CHECK(reflectionSuccessful);
        BOOST_CHECK_EQUAL(actualNumReflections, expectedNumReflections);

        hops::VectorType expectedResult = Eigen::VectorXd::Zero(endPoint.rows());

        // Algorithm is not numerically very stable for this many reflections
        BOOST_CHECK_SMALL(reflectedPoint(0) - expectedResult(0), 1e-12);
        BOOST_CHECK_SMALL(reflectedPoint(1) - expectedResult(1), 1e-12);
    }

    BOOST_AUTO_TEST_CASE(TestNoReflectionRequiredWithQuadraticConstraints) {
        Eigen::MatrixXd A(1, 1);
        A << -1;
        hops::VectorType b(1);
        b << 0;

        Eigen::MatrixXd E(1, 1);
        E << 1;
        hops::VectorType offset(1);
        offset << 0;
        double radius = 100;

        hops::VectorType startPoint(1);
        startPoint << 1;
        hops::VectorType endPoint(1);
        endPoint << 2;

        auto[reflectionSuccessful, numReflections, reflectedPoint] =
        hops::Reflector::reflectIntoPolytope(A, b, E, offset, radius, startPoint, endPoint, 200);

        BOOST_CHECK(reflectionSuccessful);
        BOOST_CHECK_EQUAL(numReflections, 0);
        BOOST_CHECK_EQUAL(reflectedPoint, endPoint);
    }

    BOOST_AUTO_TEST_CASE(Test1ReflectionStepAtLinearConstraintWithQuadraticConstraints) {
        // Constrain to positive part of real numbers, representation is Ax<b
        Eigen::MatrixXd A(1, 1);
        A << -1;
        hops::VectorType b(1);
        b << 0;

        Eigen::MatrixXd E(1, 1);
        E << 1;
        hops::VectorType offset(1);
        offset << 0;
        double radius = 100;

        hops::VectorType startPoint(1);
        startPoint << 1;
        hops::VectorType endPoint(1);
        endPoint << -9; // endpoint outside of constraints

        auto[reflectionSuccessful, numReflections, reflectedPoint] =
        hops::Reflector::reflectIntoPolytope(A, b, E, offset, radius, startPoint, endPoint, 200);

        BOOST_CHECK(reflectionSuccessful);
        BOOST_CHECK_EQUAL(numReflections, 1);
        // In this example it happens that the reflectPoint is equal to startPoint due to symmetry
        hops::VectorType expectedResult(1);
        expectedResult << 9;

        BOOST_CHECK(reflectedPoint.isApprox(expectedResult));
    }

    BOOST_AUTO_TEST_CASE(Test1ReflectionStepAtQuadraticConstraintWithQuadraticConstraints) {
        // Constrain to positive part of real numbers, representation is Ax<b
        Eigen::MatrixXd A(1, 1);
        A << -1;
        hops::VectorType b(1);
        b << 0;

        Eigen::MatrixXd E(1, 1);
        E << 1;
        hops::VectorType offset(1);
        offset << 1;
        double radius = 0.5*0.5;

        hops::VectorType startPoint(1);
        startPoint << 1;
        hops::VectorType endPoint(1);
        endPoint << 0; // endpoint outside of constraints

        auto[reflectionSuccessful, numReflections, reflectedPoint] =
        hops::Reflector::reflectIntoPolytope(A, b, E, offset, radius, startPoint, endPoint, 200);

        BOOST_CHECK(reflectionSuccessful);
        BOOST_CHECK_EQUAL(numReflections, 1);
        hops::VectorType expectedResult(1);
        expectedResult << 1.;

        BOOST_CHECK(reflectedPoint.isApprox(expectedResult));
    }

BOOST_AUTO_TEST_SUITE_END()

