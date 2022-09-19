#ifdef HOPS_CLP_FOUND

#define BOOST_TEST_MODULE LinearProgrammingClpImplTestSuite
#define BOOST_TEST_DYN_LINK

#include <boost/test/included/unit_test.hpp>
#include <Eigen/Core>
#include "hops/hops.hpp"

BOOST_AUTO_TEST_SUITE(LinearProgrammingClp)

    BOOST_AUTO_TEST_CASE(solveSmallTestProblem) {
        Eigen::VectorXd expectedParameters(2);
        expectedParameters << 8.0 / 3, 2.0 / 3;

        hops::LinearProgramSolution expectedSolution(10. / 3, expectedParameters, hops::LinearProgramStatus::OPTIMAL);

        Eigen::MatrixXd A(3, 2);
        A << 1, 2, 4, 2, -1, 1;
        Eigen::VectorXd b(3);
        b << 4, 12, 1;
        Eigen::VectorXd obj(2);
        obj << 1, 1;
        hops::LinearProgramClpImpl linearProgram(A, b);
        hops::LinearProgramSolution actualSolution = linearProgram.solve(obj);

        BOOST_CHECK_CLOSE(actualSolution.objectiveValue, expectedSolution.objectiveValue, 0.0001);
        BOOST_CHECK(actualSolution.optimalParameters.isApprox(expectedSolution.optimalParameters));
        BOOST_CHECK(actualSolution.status == expectedSolution.status);
    }

    BOOST_AUTO_TEST_CASE(computeChebyshevCenter) {
        Eigen::VectorXd expectedChebyshevParameters(2);
        expectedChebyshevParameters << 0.29289321881345, 0.29289321881345;

        Eigen::MatrixXd A(3, 2);
        Eigen::VectorXd b(3);
        A << 1, 1,
                -1, 0,
                0, -1;
        b << 1, 0, 0;

        auto linearProgram = hops::LinearProgramClpImpl(A, b);
        auto actualChebyshevCenter = linearProgram.computeChebyshevCenter();

        BOOST_CHECK(actualChebyshevCenter.optimalParameters.isApprox(expectedChebyshevParameters, 1e-11));
    }

    BOOST_AUTO_TEST_CASE(computeChebyshevCenterIsStableUnderRepeatedCalculations) {
        Eigen::MatrixXd A(3, 2);
        Eigen::VectorXd b(3);
        A << 1, 1,
                -1, 0,
                0, -1;
        b << 1, 0, 0;

        auto linearProgram = hops::LinearProgramClpImpl(A, b);
        auto actualChebyshevCenter1 = linearProgram.computeChebyshevCenter();
        auto actualChebyshevCenter2 = linearProgram.computeChebyshevCenter();
        auto actualChebyshevCenter3 = linearProgram.computeChebyshevCenter();
        auto actualChebyshevCenter4 = linearProgram.computeChebyshevCenter();

        BOOST_CHECK(actualChebyshevCenter1 == actualChebyshevCenter2);
        BOOST_CHECK(actualChebyshevCenter1 == actualChebyshevCenter3);
        BOOST_CHECK(actualChebyshevCenter1 == actualChebyshevCenter4);
        BOOST_CHECK(actualChebyshevCenter2 == actualChebyshevCenter3);
        BOOST_CHECK(actualChebyshevCenter2 == actualChebyshevCenter4);
        BOOST_CHECK(actualChebyshevCenter3 == actualChebyshevCenter4);
    }

    BOOST_AUTO_TEST_CASE(removeSingleRedundantConstraintTest) {
        Eigen::MatrixXd expectedA(4, 2);
        expectedA << 1, 0, 0, 1, -1, 0, 0, -1;
        Eigen::VectorXd expectedb(4);
        expectedb << 1, 1, 1, 1;

        Eigen::MatrixXd A(5, 2);
        A << 1, 0, 0, 1, -1, 0, 0, -1, 1, 0;
        Eigen::VectorXd b(5);
        b << 1, 1, 1, 1, 2;

        auto linearProgram = hops::LinearProgramClpImpl(A, b);
        auto[actualA, actualb] = linearProgram.removeRedundantConstraints(1e-15);

        BOOST_CHECK(actualA.isApprox(expectedA));
        BOOST_CHECK(actualb.isApprox(expectedb));
    }

    BOOST_AUTO_TEST_CASE(removeSeveralRedundantConstraintsTest) {
        Eigen::MatrixXd expectedA(4, 2);
        expectedA << 1, 0, 0, 1, -1, 0, 0, -1;
        Eigen::VectorXd expectedB(4);
        expectedB << 1, 1, 1, 1;

        Eigen::MatrixXd A(10, 2);
        A << 1, 0, 0, 1, -1, 0, 0, -1, 1, 0,
                1, 0, 1, 0, 1, 0, 1, 0, 1, 0;
        Eigen::VectorXd b(10);
        b << 1, 1, 1, 1, 2, 7, 4, 2, 100, 5;

        auto linearProgram = hops::LinearProgramClpImpl(A, b);
        auto[actualA, actualB] = linearProgram.removeRedundantConstraints(1e-15);
        BOOST_CHECK(actualA.isApprox(expectedA));
        BOOST_CHECK(actualB.isApprox(expectedB));
    }

    BOOST_AUTO_TEST_CASE(computeUnconstrainedDimensions) {
        std::vector<long> expectedUnboundDirections{1, -1};

        Eigen::MatrixXd A(2, 2);
        A << 0, 1, 0, -1;
        Eigen::VectorXd b(2);
        b << 1, 1;

        auto linearProgram = hops::LinearProgramClpImpl(A, b);
        auto actualUnboundDirections = linearProgram.computeUnconstrainedDimensions();

        BOOST_CHECK(actualUnboundDirections == expectedUnboundDirections);
    }

    BOOST_AUTO_TEST_CASE(addConstraintsToUnconstrainedDimensions) {
        Eigen::MatrixXd expectedA(4, 2);
        expectedA << 0, 1, 0, -1, 1, 0, -1, 0;
        Eigen::VectorXd expectedB(4);
        expectedB << 1, 1, 3, 2;

        Eigen::MatrixXd A(2, 2);
        A << 0, 1, 0, -1;
        Eigen::VectorXd b(2);
        b << 1, 1;

        auto linearProgram = hops::LinearProgramClpImpl(A, b);
        auto[actualA, actualB] = linearProgram.addBoxConstraintsToUnconstrainedDimensions(2, 3);

        BOOST_CHECK(actualA == expectedA);
        BOOST_CHECK(actualB == expectedB);
    }

BOOST_AUTO_TEST_SUITE_END()

#endif //HOPS_CLP_FOUND
