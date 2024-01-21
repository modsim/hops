#ifndef HOPS_LINEARTRANSFORMATION_HPP
#define HOPS_LINEARTRANSFORMATION_HPP

#include <Eigen/SVD>
#include <memory>
#include <utility>

#include "hops/Transformation/Transformation.hpp"
#include "hops/Utility/MatrixType.hpp"
#include "hops/Utility/VectorType.hpp"

namespace hops {

    /**
     * @tparam MatrixType
     * @tparam VectorType
     */
    class LinearTransformation : public Transformation {
    public:
        LinearTransformation() = default;

        LinearTransformation(const LinearTransformation &other) = default;

        LinearTransformation(const MatrixType &matrix, const VectorType &shift) : matrix(matrix), shift(shift) {}

        /**
         * @brief Transforms vector from rounded space to unrounded space.
         */
        VectorType apply(const VectorType &vector) const override {
            return matrix * vector + shift;
        }

        VectorType revert(const VectorType &vector) const override {
            if (matrix.cols() == matrix.rows() && matrix.isLowerTriangular()) {
                return matrix.template triangularView<Eigen::Lower>().solve(vector - shift);
            } else if (matrix.cols() == matrix.rows() && matrix.isUpperTriangular()) {
                return matrix.template triangularView<Eigen::Upper>().solve(vector - shift);
            } else {
                VectorType rhs = vector-shift;
                return matrix.bdcSvd(Eigen::ComputeThinU | Eigen::ComputeThinV).solve(rhs);
            }
        }

        [[nodiscard]] std::unique_ptr<Transformation> copyTransformation() const override {
            return std::make_unique<LinearTransformation>(*this);
        }

        [[nodiscard]] const MatrixType &getMatrix() const {
            return matrix;
        }

        [[nodiscard]] const VectorType &getShift() const {
            return shift;
        }

    private:
        MatrixType matrix;
        VectorType shift;
    };
}

#endif //HOPS_LINEARTRANSFORMATION_HPP

