#ifndef HOPS_MULTIVARIATEGAUSSIANMODEL_HPP
#define HOPS_MULTIVARIATEGAUSSIANMODEL_HPP

#define _USE_MATH_DEFINES
#include <math.h>
#include <Eigen/Cholesky>
#include <Eigen/Core>
#include <Eigen/LU>
#include <utility>

namespace hops {
    template<typename Matrix, typename Vector>
    class MultivariateGaussianModel {
    public:
        using MatrixType = Matrix;
        using VectorType = Vector;

        MultivariateGaussianModel(VectorType mean, MatrixType covariance);

        /**
         * @brief Evaluates the negative log likelihood for input x.
         * @param x
         * @return
         */
        typename MatrixType::Scalar calculateNegativeLogLikelihood(const VectorType &x) const;

        MatrixType calculateExpectedFisherInformation(const VectorType &) const;

        VectorType calculateLogLikelihoodGradient(const VectorType &x) const;

    private:
        VectorType mean;
        MatrixType covariance;
        MatrixType inverseCovariance;
        typename MatrixType::Scalar logNormalizationConstant;
    };

    template<typename MatrixType, typename VectorType>
    MultivariateGaussianModel<MatrixType, VectorType>::MultivariateGaussianModel(VectorType mean,
                                                                                 MatrixType covariance) :
            mean(std::move(mean)),
            covariance(std::move(covariance)) {
        Eigen::LLT<MatrixType, Eigen::Upper> solver(this->covariance);
        Eigen::Matrix<typename MatrixType::Scalar, Eigen::Dynamic, Eigen::Dynamic> matrixL = solver.matrixL();
        Eigen::Matrix<typename MatrixType::Scalar, Eigen::Dynamic, Eigen::Dynamic> matrixU = solver.matrixU();
        Eigen::Matrix<typename MatrixType::Scalar, Eigen::Dynamic, Eigen::Dynamic> inverseMatrixL = matrixL.inverse();
        inverseCovariance = inverseMatrixL * inverseMatrixL.transpose();

        logNormalizationConstant = -static_cast<typename MatrixType::Scalar>(this->mean.rows()) / 2 *
                                   std::log(2 * M_PI)
                                   - matrixL.diagonal().array().log().sum();
    }

    template<typename MatrixType, typename VectorType>
    typename MatrixType::Scalar
    MultivariateGaussianModel<MatrixType, VectorType>::calculateNegativeLogLikelihood(const VectorType &x) const {
        return -logNormalizationConstant +
               0.5 * static_cast<typename MatrixType::Scalar>((x - mean).transpose() * inverseCovariance * (x - mean));
    }

    template<typename MatrixType, typename VectorType>
    MatrixType
    MultivariateGaussianModel<MatrixType, VectorType>::calculateExpectedFisherInformation(const VectorType &) const {
        return inverseCovariance;
    }

    template<typename MatrixType, typename VectorType>
    VectorType
    MultivariateGaussianModel<MatrixType, VectorType>::calculateLogLikelihoodGradient(const VectorType &x) const {
        return -inverseCovariance * (x - mean);
    }
}

#endif //HOPS_MULTIVARIATEGAUSSIANMODEL_HPP
