#ifndef HOPS_LINEARPROGRAMFACTORY_HPP
#define HOPS_LINEARPROGRAMFACTORY_HPP

#include <Eigen/Core>
#include "LinearProgram.hpp"
#include "LinearProgramClpImpl.hpp"
#include "LinearProgramGurobiImpl.hpp"
#include <memory>

namespace hops {
    enum class LinearProgramSolver {
        CLP,
        GUROBI,
    };

    class LinearProgramFactory {
    public:
        template<typename Derived1, typename Derived2>
        static std::unique_ptr<LinearProgram> createLinearProgram(
                const Eigen::MatrixBase<Derived1> &A,
                const Eigen::MatrixBase<Derived2> &b,
                // TODO set default by checking what's available
                LinearProgramSolver solver = LinearProgramSolver::GUROBI) {
            switch (solver) {
                case LinearProgramSolver::CLP: {
                    return std::make_unique<LinearProgramClpImpl>(A, b);
                }
                default: {
                    return std::make_unique<LinearProgramGurobiImpl>(A, b);
                }
            }
        }
    };
}

#endif //HOPS_LINEARPROGRAMFACTORY_HPP
