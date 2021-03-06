#ifndef HOPS_TRANSFORMATION_HPP
#define HOPS_TRANSFORMATION_HPP

#include <utility>

namespace hops {

    /**
     * @tparam MatrixType
     * @tparam VectorType
     */
    template<typename MatrixType, typename VectorType>
    class Transformation {
    public:
        Transformation() = default;

        Transformation(MatrixType matrix, VectorType shift) : matrix(matrix), shift(shift) {}

        /**
         * @brief Transforms vector from rounded space to unrounded space.
         */
        VectorType apply(VectorType vector) {
            return matrix * vector + shift;
        }

        VectorType applyInverse(VectorType vector) {
            if (!matrix.isLowerTriangular()) {
                return matrix.inverse() * (vector - shift);
            }
            return matrix.template triangularView<Eigen::Lower>().solve(vector - shift);
        }

        MatrixType matrix;
        VectorType shift;
    };
}

#endif //HOPS_TRANSFORMATION_HPP
