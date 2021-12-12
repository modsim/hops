#ifndef HOPS_TUNING_TARGET
#define HOPS_TUNING_TARGET

#include <algorithm>
#include <memory>
#include <string>
#include <utility>

#include <hops/Utility/VectorType.hpp>

namespace hops {
    struct TuningTarget {
        virtual ~TuningTarget() = default;

        virtual std::tuple<double, double> operator()(const VectorType& x) = 0;

        virtual std::string getName() const = 0;

        virtual std::unique_ptr<TuningTarget> copyTuningTarget() const = 0;
    };
}

#endif // HOPS_TUNING_TARGET
