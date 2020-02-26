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

        virtual LinearProgramSolution solve(const Eigen::VectorXd &objective) = 0;

        /**
         * @param tolerance
         * @return A and b
         */
        virtual std::tuple<Eigen::MatrixXd, Eigen::VectorXd> removeRedundantConstraints(double tolerance) = 0;

        virtual LinearProgramSolution calculateChebyshevCenter() = 0;

        /**
         * @details dimensions with missing upper boundaries are counted starting from 1 upwards.
         *          dimensions with missing lower boundaries are counted starting from -1 downwards.
         * @return
         */
        virtual std::vector<long> calculateUnconstrainedDimensions() = 0;

        /**
         *
         * @param lb
         * @param ub
         * @return A and b
         */
        virtual std::tuple<Eigen::MatrixXd, Eigen::VectorXd> addBoxConstraintsToUnconstrainedDimensions(double lb, double ub) = 0;

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
