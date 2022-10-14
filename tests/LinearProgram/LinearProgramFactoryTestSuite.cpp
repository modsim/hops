#define BOOST_TEST_DYN_LINK
#define BOOST_TEST_MODULE LinearProgramFactoryTestSuite

#include <boost/test/unit_test.hpp>
#include "hops/hops.hpp"

BOOST_AUTO_TEST_SUITE(LinearProgramFactory)

    BOOST_AUTO_TEST_CASE(CreateClpLinearProgram) {
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

        BOOST_CHECK(solution.status == hops::LinearProgramStatus::OPTIMAL);
        BOOST_CHECK_CLOSE(solution.objectiveValue, 10.0 / 3, 0.0001);
        BOOST_CHECK(solution.optimalParameters.isApprox(expectedSolution));
#else //HOPS_CLP_FOUND
        BOOST_CHECK_THROW( {
            hops::LinearProgramFactory::createLinearProgram(A, b, hops::LinearProgramSolver::CLP); },
                           std::runtime_error );
#endif //HOPS_CLP_FOUND
    }

    BOOST_AUTO_TEST_CASE(CreateGurobiLinearProgram) {
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

        BOOST_CHECK(solution.status == hops::LinearProgramStatus::OPTIMAL);
        BOOST_CHECK_CLOSE(solution.objectiveValue, 10.0 / 3, 0.0001);
        BOOST_CHECK(solution.optimalParameters.isApprox(expectedSolution));
#else //HOPS_GUROBI_FOUND
        BOOST_CHECK_THROW( { hops::LinearProgramFactory::createLinearProgram(A, b, hops::LinearProgramSolver::GUROBI); },
                           std::runtime_error);
#endif //HOPS_GUROBI_FOUND
    }

BOOST_AUTO_TEST_SUITE_END()
