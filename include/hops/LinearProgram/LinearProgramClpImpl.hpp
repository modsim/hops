#ifndef HOPS_LINEARPROGRAMCLPIMPL_HPP
#define HOPS_LINEARPROGRAMCLPIMPL_HPP

#include <Eigen/Core>
#include "LinearProgram.hpp"

#ifdef HOPS_CLP_FOUND

#include <coin/ClpSimplex.hpp>

namespace hops {
    class LinearProgramClpImpl : public LinearProgram {
    public:
        LinearProgramClpImpl(const Eigen::MatrixXd &A, const Eigen::VectorXd &b);

        LinearProgramClpImpl(const LinearProgramClpImpl &other);

        LinearProgramClpImpl &operator=(const LinearProgramClpImpl &other);

        LinearProgramSolution solve(const Eigen::VectorXd &objective) const override;

        std::tuple<Eigen::MatrixXd, Eigen::VectorXd> removeRedundantConstraints(double tolerance) override;

        LinearProgramSolution computeChebyshevCenter() const override;

        std::vector<long> computeUnconstrainedDimensions() const override;

        std::tuple<Eigen::MatrixXd, Eigen::VectorXd>
        addBoxConstraintsToUnconstrainedDimensions(double lb, double ub) override;

    private:
        mutable ClpSimplex model;
    };
}

#else //HOPS_CLP_FOUND

namespace hops {
    class LinearProgramClpImpl : public LinearProgram {
    public:
        LinearProgramClpImpl(const Eigen::MatrixXd &A, const Eigen::VectorXd &b) : LinearProgram(A, b) {
            throw std::runtime_error("HOPS did not find CLP during compilation.");
        }

        [[nodiscard]] LinearProgramSolution solve(const Eigen::VectorXd &) const override {
            throw std::runtime_error("HOPS did not find CLP during compilation.");
        }

        std::tuple<Eigen::MatrixXd, Eigen::VectorXd> removeRedundantConstraints(double) override {
            throw std::runtime_error("HOPS did not find CLP during compilation.");
        }

        [[nodiscard]] LinearProgramSolution computeChebyshevCenter() const override {
            throw std::runtime_error("HOPS did not find CLP during compilation.");
        }

        [[nodiscard]] std::vector<long> computeUnconstrainedDimensions() const override {
            throw std::runtime_error("HOPS did not find CLP during compilation.");
        }

        std::tuple<Eigen::MatrixXd, Eigen::VectorXd>
        addBoxConstraintsToUnconstrainedDimensions(double, double) override {
            throw std::runtime_error("HOPS did not find CLP during compilation.");
        }
    };
}

#endif //HOPS_CLP_FOUND
#endif //HOPS_LINEARPROGRAMCLPIMPL_HPP
