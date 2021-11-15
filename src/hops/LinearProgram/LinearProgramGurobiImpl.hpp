#ifndef HOPS_LINEARPROGRAMGUROBIIMPL_HPP
#define HOPS_LINEARPROGRAMGUROBIIMPL_HPP

#include "LinearProgram.hpp"
#include <Eigen/Core>

#ifdef HOPS_GUROBI_FOUND

#include <gurobi_c++.h>
#include <memory>

namespace hops {
    class LinearProgramGurobiImpl : public LinearProgram {
    public:
        LinearProgramGurobiImpl(const Eigen::MatrixXd &A, const Eigen::VectorXd &b);

        LinearProgramGurobiImpl(const LinearProgramGurobiImpl &other);

        LinearProgramGurobiImpl &operator=(const LinearProgramGurobiImpl &other);

        [[nodiscard]] LinearProgramSolution solve(const Eigen::VectorXd &objective) const override;

        std::tuple<Eigen::MatrixXd, Eigen::VectorXd> removeRedundantConstraints(double tolerance) override;

        [[nodiscard]] LinearProgramSolution computeChebyshevCenter() const override;

        [[nodiscard]] std::vector<long> computeUnconstrainedDimensions() const override;

        std::tuple<Eigen::MatrixXd, Eigen::VectorXd>
        addBoxConstraintsToUnconstrainedDimensions(double lb, double ub) override;

    private:
        std::unique_ptr<GRBModel> model;
        std::vector<GRBVar> variables;
    };
}

#else //HOPS_GUROBI_FOUND

namespace hops {
    class LinearProgramGurobiImpl : public LinearProgram {
    public:
        LinearProgramGurobiImpl(const Eigen::MatrixXd &A, const Eigen::VectorXd &b) : LinearProgram(A, b) {
            throw std::runtime_error("HOPS did not find gurobi during compilation.");
        }

        [[nodiscard]] LinearProgramSolution solve(const Eigen::VectorXd &) const override {
            throw std::runtime_error("HOPS did not find gurobi during compilation.");
        }

        std::tuple<Eigen::MatrixXd, Eigen::VectorXd> removeRedundantConstraints(double) override {
            throw std::runtime_error("HOPS did not find gurobi during compilation.");
        }

        [[nodiscard]] LinearProgramSolution computeChebyshevCenter() const override {
            throw std::runtime_error("HOPS did not find gurobi during compilation.");
        }

        [[nodiscard]] std::vector<long> computeUnconstrainedDimensions() const override {
            throw std::runtime_error("HOPS did not find gurobi during compilation.");
        }

        std::tuple<Eigen::MatrixXd, Eigen::VectorXd>
        addBoxConstraintsToUnconstrainedDimensions(double, double) override {
            throw std::runtime_error("HOPS did not find gurobi during compilation.");
        }
    };
}

#endif //HOPS_GUROBI_FOUND
#endif //HOPS_LINEARPROGRAMGUROBIIMPL_HPP
