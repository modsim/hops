#ifndef HOPS_TRANSFORMATION_HPP
#define HOPS_TRANSFORMATION_HPP

#include <memory>
#include <utility>

#include "hops/Utility/MatrixType.hpp"
#include "hops/Utility/VectorType.hpp"

namespace hops {

    /**
     * @tparam MatrixType
     * @tparam VectorType
     */
    class Transformation {
    public:
        virtual ~Transformation() = default;

        virtual VectorType apply(const VectorType& vector) const { return vector; }

        virtual VectorType revert(const VectorType& vector) const { return vector; }

        virtual std::unique_ptr<Transformation> copyTransformation() const = 0;
    };
}

#endif //HOPS_TRANSFORMATION_HPP
