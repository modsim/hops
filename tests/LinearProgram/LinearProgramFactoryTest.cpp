#include <gtest/gtest.h>
#include <hops/LinearProgram/LinearProgramFactory.hpp>

namespace {
    TEST(LinearProgramFactory, CreateClpLinearProgram) {
        Eigen::MatrixXd A(3, 2);
        A << 1, 2, 4, 2, -1, 1;
        Eigen::VectorXd b(3);
        b << 4, 12, 1;
#if HOPS_CLP_FOUND
        Eigen::VectorXd expectedSolution(2);
        expectedSolution << 8.0 / 3, 2.0 / 3;

        Eigen::VectorXd obj(2);
        obj << 1, 1;

        auto linearProgram = hops::LinearProgramFactory::createLinearProgram(A, b, hops::LinearProgramSolver::CLP);
        auto solution = linearProgram->solve(obj);

        EXPECT_EQ(solution.status, hops::LinearProgramStatus::OPTIMAL);
        EXPECT_NEAR(solution.objectiveValue, 10.0 / 3, 0.0001);
        EXPECT_TRUE(solution.optimalParameters.isApprox(expectedSolution));
#else //HOPS_CLP_FOUND
        EXPECT_ANY_THROW(hops::LinearProgramFactory::createLinearProgram(A, b, hops::LinearProgramSolver::CLP));
#endif //HOPS_CLP_FOUND
    }

    TEST(LinearProgramFactory, CreateGurobiLinearProgram) {
        Eigen::MatrixXd A(3, 2);
        A << 1, 2, 4, 2, -1, 1;
        Eigen::VectorXd b(3);
        b << 4, 12, 1;
#if HOPS_GUROBI_FOUND
        Eigen::VectorXd expectedSolution(2);
        expectedSolution << 8.0 / 3, 2.0 / 3;

        Eigen::VectorXd obj(2);
        obj << 1, 1;

        auto linearProgram = hops::LinearProgramFactory::createLinearProgram(A, b, hops::LinearProgramSolver::GUROBI);
        auto solution = linearProgram->solve(obj);

        EXPECT_EQ(solution.status, hops::LinearProgramStatus::OPTIMAL);
        EXPECT_NEAR(solution.objectiveValue, 10.0 / 3, 0.0001);
        EXPECT_TRUE(solution.optimalParameters.isApprox(expectedSolution));
#else //HOPS_GUROBI_FOUND
        EXPECT_ANY_THROW(hops::LinearProgramFactory::createLinearProgram(A, b, hops::LinearProgramSolver::GUROBI));
#endif //HOPS_GUROBI_FOUND
    }
}
