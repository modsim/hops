#ifdef HOPS_GUROBI_FOUND

#include <Eigen/Core>
#include <gtest/gtest.h>
#include <hops/LinearProgram/LinearProgramSolution.hpp>
#include <hops/LinearProgram/LinearProgramGurobiImpl.hpp>

namespace {
    TEST(LinearProgrammingGurobiImpl, solveSmallTestProblem) {
        Eigen::VectorXd expectedParameters(2);
        expectedParameters << 8.0 / 3, 2.0 / 3;

        hops::LinearProgramSolution expectedSolution(10. / 3, expectedParameters, hops::LinearProgramStatus::OPTIMAL);

        Eigen::MatrixXd A(3, 2);
        A << 1, 2, 4, 2, -1, 1;
        Eigen::VectorXd b(3);
        b << 4, 12, 1;
        Eigen::VectorXd obj(2);
        obj << 1, 1;
        hops::LinearProgramGurobiImpl linearProgram(A, b);
        hops::LinearProgramSolution actualSolution = linearProgram.solve(obj);

        EXPECT_NEAR(actualSolution.objectiveValue, expectedSolution.objectiveValue, 0.0001);
        EXPECT_TRUE(actualSolution.optimalParameters.isApprox(expectedSolution.optimalParameters));
        EXPECT_EQ(actualSolution.status, expectedSolution.status);
    }

    TEST(LinearProgrammingGurobiImpl, calculateChebyshevCenter) {
        Eigen::VectorXd expectedChebyshevCenter(2);
        expectedChebyshevCenter << 0.29289321881345, 0.29289321881345;

        Eigen::MatrixXd A(3, 2);
        Eigen::VectorXd b(3);
        A << 1, 1,
                -1, 0,
                0, -1;
        b << 1, 0, 0;

        auto linearProgram = hops::LinearProgramGurobiImpl(A, b);
        auto actualChebyshevCenter = linearProgram.calculateChebyshevCenter();

        EXPECT_TRUE(actualChebyshevCenter.optimalParameters.isApprox(expectedChebyshevCenter, 1e-12));
    }

    TEST(LinearProgrammingGurobiImpl, calculateChebyshevCenterIsStableUnderRepeatedCalculations) {
        Eigen::MatrixXd A(3, 2);
        Eigen::VectorXd b(3);
        A << 1, 1,
                -1, 0,
                0, -1;
        b << 1, 0, 0;

        auto linearProgram = hops::LinearProgramGurobiImpl(A, b);
        auto actualChebyshevCenter1 = linearProgram.calculateChebyshevCenter();
        auto actualChebyshevCenter2 = linearProgram.calculateChebyshevCenter();
        auto actualChebyshevCenter3 = linearProgram.calculateChebyshevCenter();
        auto actualChebyshevCenter4 = linearProgram.calculateChebyshevCenter();

        EXPECT_EQ(actualChebyshevCenter1, actualChebyshevCenter2);
        EXPECT_EQ(actualChebyshevCenter1, actualChebyshevCenter3);
        EXPECT_EQ(actualChebyshevCenter1, actualChebyshevCenter4);
        EXPECT_EQ(actualChebyshevCenter2, actualChebyshevCenter3);
        EXPECT_EQ(actualChebyshevCenter2, actualChebyshevCenter4);
        EXPECT_EQ(actualChebyshevCenter3, actualChebyshevCenter4);
    }

    TEST(LinearProgrammingGurobiImpl, removeSingleRedundantConstraintTest) {
        Eigen::MatrixXd expectedA(4, 2);
        expectedA << 1, 0, 0, 1, -1, 0, 0, -1;
        Eigen::VectorXd expectedb(4);
        expectedb << 1, 1, 1, 1;

        Eigen::MatrixXd A(5, 2);
        A << 1, 0, 0, 1, -1, 0, 0, -1, 1, 0;
        Eigen::VectorXd b(5);
        b << 1, 1, 1, 1, 2;

        auto linearProgram = hops::LinearProgramGurobiImpl(A, b);
        auto[actualA, actualb] = linearProgram.removeRedundantConstraints(1e-15);

        EXPECT_TRUE(actualA.isApprox(expectedA));
        EXPECT_TRUE(actualb.isApprox(expectedb));
    }

    TEST(LinearProgrammingGurobiImpl, removeSeveralRedundantConstraintsTest) {
        Eigen::MatrixXd expectedA(4, 2);
        expectedA << 1, 0, 0, 1, -1, 0, 0, -1;
        Eigen::VectorXd expectedB(4);
        expectedB << 1, 1, 1, 1;

        Eigen::MatrixXd A(10, 2);
        A << 1, 0, 0, 1, -1, 0, 0, -1, 1, 0,
                1, 0, 1, 0, 1, 0, 1, 0, 1, 0;
        Eigen::VectorXd b(10);
        b << 1, 1, 1, 1, 2, 7, 4, 2, 100, 5;

        auto linearProgram = hops::LinearProgramGurobiImpl(A, b);
        auto[actualA, actualB] = linearProgram.removeRedundantConstraints(1e-15);
        EXPECT_TRUE(actualA.isApprox(expectedA));
        EXPECT_TRUE(actualB.isApprox(expectedB));
    }

    TEST(LinearProgrammingGurobiImpl, calculateUnconstrainedDimensions) {
        std::vector<long> expectedUnboundDirections{1, -1};

        Eigen::MatrixXd A(2, 2);
        A << 0, 1, 0, -1;
        Eigen::VectorXd b(2);
        b << 1, 1;

        auto linearProgram = hops::LinearProgramGurobiImpl(A, b);
        auto actualUnboundDirections = linearProgram.calculateUnconstrainedDimensions();

        EXPECT_EQ(actualUnboundDirections, expectedUnboundDirections);
    }

    TEST(LinearProgrammingGurobiImpl, addConstraintsToUnconstrainedDimensions) {
        Eigen::MatrixXd expectedA(4, 2);
        expectedA << 0, 1, 0, -1, 1, 0, -1, 0;
        Eigen::VectorXd expectedB(4);
        expectedB << 1, 1, 3, 2;

        Eigen::MatrixXd A(2, 2);
        A << 0, 1, 0, -1;
        Eigen::VectorXd b(2);
        b << 1, 1;

        auto linearProgram = hops::LinearProgramGurobiImpl(A, b);
        auto[actualA, actualB] = linearProgram.addBoxConstraintsToUnconstrainedDimensions(2, 3);

        EXPECT_EQ(actualA, expectedA);
        EXPECT_EQ(actualB , expectedB);
    }
}

#endif //HOPS_GUROBI_FOUND
