#ifndef HOPS_GRIDSEARCHTUNER_HPP
#define HOPS_GRIDSEARCHTUNER_HPP

#include "hops/Utility/MatrixType.hpp"
#include "hops/Utility/VectorType.hpp"

#include <Eigen/Core>

#include <chrono>
#include <cmath>
#include <memory>
#include <stdexcept>

namespace hops {
    class GridSearchTuner {
    public:
        //TODO
        struct param_type {
            param_type() = delete;
        };

        GridSearchTuner() = delete;
    };
} // namespace hops

#endif // HOPS_GRIDSEARCHTUNER_HPP
