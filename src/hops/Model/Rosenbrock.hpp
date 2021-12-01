#ifndef HOPS_ROSENBROCK_HPP
#define HOPS_ROSENBROCK_HPP

#include <cmath>
#include <stdexcept>
#include <unsupported/Eigen/MatrixFunctions>
#include <utility>

#include <hops/Model/Model.hpp>
#include <hops/Utility/MatrixType.hpp>
#include <hops/Utility/VectorType.hpp>

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

        [[nodiscard]] typename MatrixType::Scalar computeNegativeLogLikelihood(const VectorType &x) const override;

        [[nodiscard]] MatrixType computeHessian(const VectorType &x) const;

        [[nodiscard]] std::optional<VectorType> computeLogLikelihoodGradient(const VectorType &x) const override;

        /**
         * @brief Actually this computes the softmax of the hessian instead of the
         * the expected fisher information is intractable for this model.
         * @details See 10.1007/978-3-642-40020-9_35
         * @return
         */
        [[nodiscard]] std::optional<MatrixType> computeExpectedFisherInformation(const VectorType &x) const override;

        double getScaleParameter() const;

        [[nodiscard]] const VectorType &getShiftParameter() const;

        long getNumberOfDimensions() const;

        [[nodiscard]] std::unique_ptr<Model> deepCopy() const override;

    private:
        typename MatrixType::Scalar scaleParameter;
        VectorType shiftParameter;
        long numberOfDimensions;
    };

    Rosenbrock::Rosenbrock(double scaleParameter,
                           VectorType shiftParameter) :
            scaleParameter(scaleParameter),
            shiftParameter(std::move(shiftParameter)) {
        numberOfDimensions = this->shiftParameter.rows() * 2;
    }

    typename MatrixType::Scalar
    Rosenbrock::computeNegativeLogLikelihood(const VectorType &x) const {
        if (x.rows() != numberOfDimensions) {
            throw std::runtime_error("Input x has wrong number of rows.");
        }
        typename MatrixType::Scalar result = 0;
        for (long i = 0; i < shiftParameter.rows(); ++i) {
            result += scaleParameter *
                      (100 * std::pow(std::pow(x(2 * i), 2) - x(2 * i + 1), 2) +
                       std::pow(x(2 * i) - shiftParameter(i), 2));
        }
        return result;
    }

    MatrixType
    Rosenbrock::computeHessian(const VectorType &x) const {
        MatrixType hessian = MatrixType::Zero(x.rows(), x.rows());

        for (long i = 0; i < shiftParameter.rows(); ++i) {
            hessian(2 * i, 2 * i) =
                    scaleParameter * (1200 * std::pow(x(2 * i), 2) - 400 * x(2 * i + 1) + 2);
            hessian(2 * i + 1, 2 * i) = scaleParameter * -400 * x(2 * i);
            hessian(2 * i, 2 * i + 1) = scaleParameter * -400 * x(2 * i);
            hessian(2 * i + 1, 2 * i + 1) = scaleParameter * 200;
        }

        return hessian;
    }

    std::optional<VectorType>
    Rosenbrock::computeLogLikelihoodGradient(const VectorType &x) const {
        VectorType gradient = VectorType::Zero(x.rows());
        for (long i = 0; i < shiftParameter.rows(); ++i) {
            gradient(2 * i) = scaleParameter * (4 * 100 * (x(2 * i + 1) - std::pow(x(2 * i), 2)) * (-2 * x(2 * i)) +
                                                2 * (x(2 * i) - shiftParameter(i)));

            gradient(2 * i + 1) = scaleParameter * 2 * 100 * (x(2 * i + 1) - std::pow(x(2 * i), 2));
        }
        return gradient;
    }

    std::optional<MatrixType>
    Rosenbrock::computeExpectedFisherInformation(const VectorType &x) const {
        MatrixType hessian = computeHessian(x);
        // regularization should be between 0 and infinity. This value is guessed for now.
        double regularization = 1. / hessian.maxCoeff();
        MatrixType expPositive = (regularization * hessian).exp();
        MatrixType expNegative = (-regularization * hessian).exp();
        hessian = (expPositive + expNegative) * hessian * (expPositive - expNegative).inverse();
        return
                hessian;
    }

    std::unique_ptr<Model> Rosenbrock::deepCopy() const {
        return std::make_unique<Rosenbrock>(scaleParameter, shiftParameter);
    }

    double Rosenbrock::getScaleParameter() const {
        return scaleParameter;
    }

    const VectorType &Rosenbrock::getShiftParameter() const {
        return shiftParameter;
    }

    long Rosenbrock::getNumberOfDimensions() const {
        return numberOfDimensions;
    }
}

#endif //HOPS_ROSENBROCK_HPP
