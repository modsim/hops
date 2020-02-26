#ifndef HOPS_LINEARPROGRAMSOLUTION_HPP
#define HOPS_LINEARPROGRAMSOLUTION_HPP

#include <Eigen/Core>
#include <utility>
#include "LinearProgramStatus.hpp"

namespace hops {
    struct LinearProgramSolution {
        LinearProgramSolution(double objectiveValue, Eigen::VectorXd solution, LinearProgramStatus status)
                : objectiveValue(objectiveValue), optimalParameters(std::move(solution)), status(status) {}

        bool operator==(const LinearProgramSolution &rhs) const {
            return objectiveValue == rhs.objectiveValue &&
                   optimalParameters == rhs.optimalParameters &&
                   status == rhs.status;
        }

        bool operator!=(const LinearProgramSolution &rhs) const {
            return !(rhs == *this);
        }

        double objectiveValue;
        Eigen::VectorXd optimalParameters;
        LinearProgramStatus status;
    };
}

#endif //HOPS_LINEARPROGRAMSOLUTION_HPP
