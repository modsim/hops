#ifdef HOPS_GUROBI_FOUND

#ifndef HOPS_LINEARPROGRAMGUROBIIMPL_HPP
#define HOPS_LINEARPROGRAMGUROBIIMPL_HPP

#include <Eigen/Core>
#include <gurobi_c++.h>
#include <memory>
#include "LinearProgram.hpp"

namespace hops {
    class LinearProgramGurobiImpl : public LinearProgram {
    public:
        LinearProgramGurobiImpl(const Eigen::MatrixXd &A, Eigen::VectorXd b);

        LinearProgramGurobiImpl(const LinearProgramGurobiImpl &other);

        LinearProgramGurobiImpl &operator=(const LinearProgramGurobiImpl &other);

        LinearProgramSolution solve(const Eigen::VectorXd &objective) const override;

        std::tuple<Eigen::MatrixXd, Eigen::VectorXd> removeRedundantConstraints(double tolerance) override;

        LinearProgramSolution calculateChebyshevCenter() const override;

        std::vector<long> calculateUnconstrainedDimensions() const override;

        std::tuple<Eigen::MatrixXd, Eigen::VectorXd>
        addBoxConstraintsToUnconstrainedDimensions(double lb, double ub) override;

    private:
        std::unique_ptr<GRBModel> model;
        std::vector<GRBVar> variables;
    };
}

#endif //HOPS_LINEARPROGRAMGUROBIIMPL_HPP
#endif //HOPS_GUROBI_FOUND
