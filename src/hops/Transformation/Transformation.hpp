#ifndef HOPS_TRANSFORMATION_HPP
#define HOPS_TRANSFORMATION_HPP

#include <utility>

#include <hops/Utility/MatrixType.hpp>
#include <hops/Utility/VectorType.hpp>

namespace hops {

    /**
     * @tparam MatrixType
     * @tparam VectorType
     */
    class Transformation {
    public:
        virtual VectorType apply(const VectorType& vector) const { return vector; }
        virtual VectorType revert(const VectorType& vector) const { return vector; }
    };
}

#endif //HOPS_TRANSFORMATION_HPP
