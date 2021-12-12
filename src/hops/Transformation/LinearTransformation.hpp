#ifndef HOPS_LINEARTRANSFORMATION_HPP
#define HOPS_LINEARTRANSFORMATION_HPP

#include <memory>
#include <utility>

#include <hops/Transformation/Transformation.hpp>
#include <hops/Utility/MatrixType.hpp>
#include <hops/Utility/VectorType.hpp>

namespace hops {

    /**
     * @tparam MatrixType
     * @tparam VectorType
     */
    class LinearTransformation : public Transformation {
    public:
        LinearTransformation() = default;

        LinearTransformation(const LinearTransformation& other) = default;

        LinearTransformation(const MatrixType &matrix, const VectorType &shift) : matrix(matrix), shift(shift) {}

        /**
         * @brief Transforms vector from rounded space to unrounded space.
         */
        VectorType apply(const VectorType &vector) const override {
            return matrix * vector + shift;
        }

        VectorType revert(const VectorType &vector) const override {
            if (!matrix.isLowerTriangular()) {
                return matrix.inverse() * (vector - shift);
            }
            return matrix.template triangularView<Eigen::Lower>().solve(vector - shift);
        }

        std::unique_ptr<Transformation> copyTransformation() const override {
            return std::make_unique<LinearTransformation>(*this);
        }

    private:
        MatrixType matrix;
        VectorType shift;
    };
}

#endif //HOPS_LINEARTRANSFORMATION_HPP

