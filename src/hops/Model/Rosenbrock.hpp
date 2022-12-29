#ifndef HOPS_ROSENBROCK_HPP
#define HOPS_ROSENBROCK_HPP

#include <cmath>
#include <stdexcept>
#include <unsupported/Eigen/MatrixFunctions>
#include <utility>

#include "hops/Model/Model.hpp"
#include "hops/Utility/MatrixType.hpp"
#include "hops/Utility/VectorType.hpp"

namespace hops {

    /**
     * @brief multi-dimensional extension of rosenbrock function to N dimensions. Only defined on spaces of even N!
     * @details Reference: https://doi.org/10.1162/evco.2009.17.3.437 \n
     *  Definition: \f$ f(x_1, x_2,..., x_N) = \sum_{i=1}^{N/2} [s_i \cdot (x^2_{2i-1}-x_{2i})^2 + (x_{2i-1} -a_i)^2] \f$ \n
     *  where \f$ \boldsymbol{a} \f$ is the shiftParameter and \f$ \boldsymbol{s} \f$ is the scaleParameter in the Constructor. Both vectors
     *  have dimensions (\f$ \frac{N}{2} \f$).
     */
    class Rosenbrock : public Model {
    public:
        /**
         * @Brief shiftParameter has half the dimensions of the state vector
         * @param scaleParameter
         * @param shiftParameter
         */
        Rosenbrock(double scaleParameter, VectorType shiftParameter);

        [[nodiscard]] typename MatrixType::Scalar computeNegativeLogLikelihood(const VectorType &x) override;

        [[nodiscard]] MatrixType computeHessian(const VectorType &x);

        [[nodiscard]] std::optional<VectorType> computeLogLikelihoodGradient(const VectorType &x) override;

        /**
         * @brief Actually this computes the softmax of the hessian instead of the
         * the expected fisher information is intractable for this model.
         * @details See 10.1007/978-3-642-40020-9_35
         * @return
         */
        [[nodiscard]] std::optional<MatrixType> computeExpectedFisherInformation(const VectorType &x) override;

        double getScaleParameter() const;

        [[nodiscard]] const VectorType &getShiftParameter() const;

        long getNumberOfDimensions() const;

        [[nodiscard]] std::unique_ptr<Model> copyModel() const override;

        std::vector<std::string> getDimensionNames() const override;

    private:
        typename MatrixType::Scalar scaleParameter;
        VectorType shiftParameter;
        long numberOfDimensions;
    };
}

#endif //HOPS_ROSENBROCK_HPP
