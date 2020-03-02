#ifdef HOPS_CLP_FOUND

#ifndef HOPS_LINEARPROGRAMCLPIMPL_HPP
#define HOPS_LINEARPROGRAMCLPIMPL_HPP

#include <coin/ClpSimplex.hpp>
#include <Eigen/Core>
#include "LinearProgram.hpp"

namespace hops {
    class LinearProgramClpImpl : public LinearProgram {
    public:
        LinearProgramClpImpl(const Eigen::MatrixXd &A, const Eigen::VectorXd &b);

        LinearProgramClpImpl(const LinearProgramClpImpl &other);

        LinearProgramClpImpl &operator=(const LinearProgramClpImpl &other);

        LinearProgramSolution solve(const Eigen::VectorXd &objective) const override;

        std::tuple<Eigen::MatrixXd, Eigen::VectorXd> removeRedundantConstraints(double tolerance) override;

        LinearProgramSolution calculateChebyshevCenter() const override;

        std::vector<long> calculateUnconstrainedDimensions() const override;

        std::tuple<Eigen::MatrixXd, Eigen::VectorXd>
        addBoxConstraintsToUnconstrainedDimensions(double lb, double ub) override;

    private:
        mutable ClpSimplex model;
    };
}


#endif //HOPS_LINEARPROGRAMCLPIMPL_HPP
#endif //HOPS_CLP_FOUND
