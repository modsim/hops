#ifndef NUPS_TRANSFORMATION_HPP
#define NUPS_TRANSFORMATION_HPP

#include <utility>

namespace nups {

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

        MatrixType matrix;
        VectorType shift;
    };
}

#endif //NUPS_TRANSFORMATION_HPP
