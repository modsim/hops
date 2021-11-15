#ifndef HOPS_LINEARPROGRAMMING_HPP
#define HOPS_LINEARPROGRAMMING_HPP

#include <Eigen/Core>
#include "LinearProgramSolution.hpp"
#include <utility>

namespace hops {
    class LinearProgram {
    public:
        LinearProgram(Eigen::MatrixXd a, Eigen::VectorXd b) : A(std::move(a)), b(std::move(b)) {}

        virtual ~LinearProgram() = default;

        [[nodiscard]] virtual LinearProgramSolution solve(const Eigen::VectorXd &objective) const = 0;

        /**
         * @brief Removes redundant constraints and returns system matrices. Changes to the system matrices
         *        are reflected internally in the LP solver.
         * @param tolerance
         * @return A and b
         */
        virtual std::tuple<Eigen::MatrixXd, Eigen::VectorXd> removeRedundantConstraints(double tolerance) = 0;

        [[nodiscard]] virtual LinearProgramSolution computeChebyshevCenter() const = 0;

        /**
         * @details dimensions with missing upper boundaries are counted starting from 1 upwards.
         *          dimensions with missing lower boundaries are counted starting from -1 downwards.
         * @return
         */
        [[nodiscard]] virtual std::vector<long> computeUnconstrainedDimensions() const = 0;

        /**
         * @brief Adds box constraints to unconstrained dimensions and returns system matrices.
         *        Changes to the system matrices are reflected internally in the LP solver.
         * @param lb
         * @param ub
         * @return A and b
         */
        virtual std::tuple<Eigen::MatrixXd, Eigen::VectorXd>
        addBoxConstraintsToUnconstrainedDimensions(double lb, double ub) = 0;

        [[nodiscard]] const Eigen::MatrixXd &getA() const {
            return A;
        }

        [[nodiscard]] const Eigen::VectorXd &getB() const {
            return b;
        }

    protected:
        Eigen::MatrixXd A;
        Eigen::VectorXd b;
    };
}

#endif //HOPS_LINEARPROGRAMMING_HPP
