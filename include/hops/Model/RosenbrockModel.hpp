#ifndef HOPS_ROSENBROCKMODEL_HPP
#define HOPS_ROSENBROCKMODEL_HPP

#include <cmath>
#include <stdexcept>
#include <unsupported/Eigen/MatrixFunctions>
#include <utility>

namespace hops {

    /**
     * @brief multi-dimensional extension of rosenbrock funtion to N dimensions. Only defined on spaces of even N.
     * @details Reference: https://doi.org/10.1162/evco.2009.17.3.437 \n
     *  Definition: \f$ f(x_1, x_2,..., x_N) = \sum_{i=1}^{N/2} [s_i \cdot (x^2_{2i-1}-x_{2i})^2 + (x_{2i-1} -a_i)^2] \f$ \n
     *  where \f$ \boldsymbol{a} \f$ is the shiftParameter and \f$ \boldsymbol{s} \f$ is the scaleParameter in the Constructor. Both vectors
     *  have dimensions (\f$ \frac{N}{2} \f$).
     */
    template<typename Matrix=Eigen::MatrixXd, typename Vector=Eigen::VectorXd>
    class RosenbrockModel {
    public:
        using MatrixType = Matrix;
        using VectorType = Vector;

        RosenbrockModel(double scaleParameter, VectorType shiftParameter);

        typename MatrixType::Scalar calculateNegativeLogLikelihood(const VectorType &x) const;

        MatrixType calculateHessian(const VectorType &x) const;

        VectorType calculateLogLikelihoodGradient(const VectorType &x) const;

        /**
         * @brief Actually this calculates the softmax of the hessian instead of the
         * the expected fisher information is intractable for this model.
         * @details See 10.1007/978-3-642-40020-9_35
         * @return
         */
        MatrixType calculateExpectedFisherInformation(const VectorType &x) const {
            MatrixType hessian = calculateHessian(x);
//            // regularization should be between 0 and infinity. This value is guessed for now.
//            double regularization = 1. / hessian.maxCoeff();
//            MatrixType expPositive = (regularization * hessian).exp();
//            MatrixType expNegative = (-regularization * hessian).exp();
//            hessian = (expPositive + expNegative) * hessian * (expPositive - expNegative).inverse();
            return hessian;
        }


    private:
        typename MatrixType::Scalar scaleParameter;
        VectorType shiftParameter;
        long numberOfDimensions;
    };

    template<typename Matrix, typename Vector>
    RosenbrockModel<Matrix, Vector>::RosenbrockModel(double scaleParameter,
                                                     VectorType shiftParameter) :
            scaleParameter(scaleParameter),
            shiftParameter(std::move(shiftParameter)) {
        numberOfDimensions = this->shiftParameter.rows() * 2;
    }

    template<typename MatrixType, typename VectorType>
    typename MatrixType::Scalar
    RosenbrockModel<MatrixType, VectorType>::calculateNegativeLogLikelihood(const VectorType &x) const {
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

    template<typename MatrixType, typename VectorType>
    MatrixType
    RosenbrockModel<MatrixType, VectorType>::calculateHessian(const VectorType &x) const {
        MatrixType observedFisherInformation(x.rows(), x.rows());

        for (long i = 0; i < shiftParameter.rows(); ++i) {
            observedFisherInformation(2 * i, 2 * i) =
                    scaleParameter * (1200 * std::pow(x(2 * i), 2) - 400 * x(2 * i + 1) + 2);
            observedFisherInformation(2 * i + 1, 2 * i) = scaleParameter * -400 * x(2 * i);
            observedFisherInformation(2 * i, 2 * i + 1) = scaleParameter * -400 * x(2 * i);
            observedFisherInformation(2 * i + 1, 2 * i + 1) = scaleParameter * 200;
        }

        return observedFisherInformation;
    }

    template<typename MatrixType, typename VectorType>
    VectorType
    RosenbrockModel<MatrixType, VectorType>::calculateLogLikelihoodGradient(const VectorType &x) const {
        VectorType gradient(x.rows());
        for (long i = 0; i < shiftParameter.rows(); ++i) {
            gradient(2 * i) = scaleParameter * (4 * 100 * (x(2 * i + 1) - std::pow(x(2 * i), 2)) * (-2 * x(2 * i)) +
                                                2 * (x(2 * i) - shiftParameter(i)));

            gradient(2 * i + 1) = scaleParameter * 2 * 100 * (x(2 * i + 1) - std::pow(x(2 * i), 2));
        }
        return gradient;
    }
}

#endif //HOPS_ROSENBROCKMODEL_HPP
