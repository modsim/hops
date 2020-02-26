#ifndef HOPS_LINEARPROGRAMGUROBIIMPL_HPP
#define HOPS_LINEARPROGRAMGUROBIIMPL_HPP

#include <Eigen/Core>
#include <gurobi_c++.h>
#include "hops/LinearProgram/LinearProgram.hpp"

namespace hops {
    class LinearProgramGurobiImpl : public LinearProgram {
    public:
        LinearProgramGurobiImpl(Eigen::MatrixXd A, Eigen::VectorXd b);

        LinearProgramGurobiImpl(const LinearProgramGurobiImpl& other);

        LinearProgramGurobiImpl operator=(const LinearProgramGurobiImpl& other);

        LinearProgramSolution solve(const Eigen::VectorXd &objective) override;

        std::tuple<Eigen::MatrixXd, Eigen::VectorXd> removeRedundantConstraints(double tolerance) override;

        LinearProgramSolution calculateChebyshevCenter() override;

        std::vector<long> calculateUnconstrainedDimensions() override;

        std::tuple<Eigen::MatrixXd, Eigen::VectorXd>
        addBoxConstraintsToUnconstrainedDimensions(double lb, double ub) override;

    private:
        GRBModel model;
        std::vector<GRBVar> variables;
    };
}

#endif //HOPS_LINEARPROGRAMGUROBIIMPL_HPP
