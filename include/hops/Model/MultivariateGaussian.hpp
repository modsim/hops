#ifndef HOPS_MULTIVARIATEGAUSSIAN_HPP
#define HOPS_MULTIVARIATEGAUSSIAN_HPP

#define _USE_MATH_DEFINES
#include <math.h> // Using deprecated math for windows
#include <Eigen/Cholesky>
#include <Eigen/Core>
#include <Eigen/LU>
#include <utility>
#include <hops/Model/Model.hpp>

namespace hops {
    class MultivariateGaussian : public Model {
    public:
        MultivariateGaussian(VectorType mean, MatrixType covariance);

        /**
         * @brief Evaluates the negative log likelihood for input x.
         * @param x
         * @return
         */
        [[nodiscard]] MatrixType::Scalar computeNegativeLogLikelihood(const VectorType &x) const override;

        [[nodiscard]] std::optional<VectorType> computeLogLikelihoodGradient(const VectorType &x) const override;

        [[nodiscard]] std::optional<MatrixType> computeExpectedFisherInformation(const VectorType &) const override;

        [[nodiscard]] const VectorType &getMean() const;

        [[nodiscard]] const MatrixType &getCovariance() const;

    private:
        VectorType mean;
        MatrixType covariance;
        MatrixType inverseCovariance;
        typename MatrixType::Scalar logNormalizationConstant;
    };

    MultivariateGaussian::MultivariateGaussian(VectorType mean, MatrixType covariance) :
            mean(std::move(mean)),
            covariance(std::move(covariance)) {
        Eigen::LLT<MatrixType> solver(this->covariance);
        if (!this->covariance.isApprox(this->covariance.transpose()) || solver.info() == Eigen::NumericalIssue) {
            throw std::domain_error(
                    "Possibly non semi-positive definite covariance in initialization for MultivariateGaussian.");
        }
        Eigen::Matrix<typename MatrixType::Scalar, Eigen::Dynamic, Eigen::Dynamic> matrixL = solver.matrixL();
        Eigen::Matrix<typename MatrixType::Scalar, Eigen::Dynamic, Eigen::Dynamic> inverseMatrixL = matrixL.inverse();
        inverseCovariance = inverseMatrixL.transpose() * inverseMatrixL;

        logNormalizationConstant = -static_cast<typename MatrixType::Scalar>(this->mean.rows()) / 2 *
                                   std::log(2 * M_PI)
                                   - matrixL.diagonal().array().log().sum();
    }

    typename MatrixType::Scalar
    MultivariateGaussian::computeNegativeLogLikelihood(const VectorType &x) const {
        return -logNormalizationConstant +
               0.5 * static_cast<typename MatrixType::Scalar>((x - mean).transpose() * inverseCovariance * (x - mean));
    }

    std::optional<VectorType> MultivariateGaussian::computeLogLikelihoodGradient(const VectorType &x) const {
        return -inverseCovariance * (x - mean);
    }

    std::optional<MatrixType> MultivariateGaussian::computeExpectedFisherInformation(const VectorType &) const {
        return inverseCovariance;
    }

    const VectorType &MultivariateGaussian::getMean() const {
        return mean;
    }

    const MatrixType &MultivariateGaussian::getCovariance() const {
        return covariance;
    }
}

#endif //HOPS_MULTIVARIATEGAUSSIAN_HPP
