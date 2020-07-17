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

#if defined(HOPS_CLP_FOUND) && defined(HOPS_GUROBI_FOUND)

    class LinearProgramFactory {
    public:
        template<typename Derived1, typename Derived2>
        static std::unique_ptr<LinearProgram> createLinearProgram(
                const Eigen::MatrixBase<Derived1> &A,
                const Eigen::MatrixBase<Derived2> &b,
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

#elif defined(HOPS_CLP_FOUND)

    class LinearProgramFactory {
    public:
        template<typename Derived1, typename Derived2>
        static std::unique_ptr<LinearProgram> createLinearProgram(
                const Eigen::MatrixBase<Derived1> &A,
                const Eigen::MatrixBase<Derived2> &b,
                LinearProgramSolver solver = LinearProgramSolver::CLP) {
            switch (solver) {
                case LinearProgramSolver::CLP: {
                    return std::make_unique<LinearProgramClpImpl>(A, b);
                }
                default: {
                    throw std::runtime_error("Linear program solver was not found.");
                }
            }
        }
    };

#elif defined(HOPS_GUROBI_FOUND)

    class LinearProgramFactory {
    public:
        template<typename Derived1, typename Derived2>
        static std::unique_ptr<LinearProgram> createLinearProgram(
                const Eigen::MatrixBase<Derived1> &A,
                const Eigen::MatrixBase<Derived2> &b,
                LinearProgramSolver solver = LinearProgramSolver::GUROBI) {
            switch (solver) {
                case LinearProgramSolver::GUROBI: {
                    return std::make_unique<LinearProgramGurobiImpl>(A, b);
                }
                default: {
                    throw std::runtime_error("Linear program solver was not found.");
                }
            }
        }
    };

#else

    class LinearProgramFactory {
    public:
        template<typename Derived1, typename Derived2>
        static std::unique_ptr<LinearProgram> createLinearProgram(
                const Eigen::MatrixBase<Derived1> &A,
                const Eigen::MatrixBase<Derived2> &b,
                LinearProgramSolver solver = LinearProgramSolver::CLP) {
            (void)A;
            (void)b;
            (void)solver;
            throw std::runtime_error("No linear program solver was found.");
        }
    };

#endif //HOPS_CLP_FOUND && HOPS_GUROBI_FOUND

}

#endif //HOPS_LINEARPROGRAMFACTORY_HPP
