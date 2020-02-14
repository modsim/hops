#ifndef HOPS_MULTIVARIATEGAUSSIANMODEL_HPP
#define HOPS_MULTIVARIATEGAUSSIANMODEL_HPP

#include <boost/math/constants/constants.hpp>
#include <cmath>
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
        double calculateNegativeLogLikelihood(const VectorType &x) const;

        MatrixType calculateExpectedFisherInformation(const VectorType &) const;

        VectorType calculateGradient(const VectorType &x) const;

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
            covariance(std::move(covariance)),
            inverseCovariance(this->covariance.inverse()) {
        Eigen::LLT<MatrixType, Eigen::Upper> solver(this->covariance);
        logNormalizationConstant =
                -std::log(std::sqrt(std::pow(2 * boost::math::constants::pi<double>(), this->mean.rows())))
                - MatrixType(solver.matrixU()).diagonal().array().log().sum();
    }

    template<typename MatrixType, typename VectorType>
    double
    MultivariateGaussianModel<MatrixType, VectorType>::calculateNegativeLogLikelihood(const VectorType &x) const {
        return -logNormalizationConstant +
               0.5 * static_cast<double>((x - mean).transpose() * inverseCovariance * (x - mean));
    }

    template<typename MatrixType, typename VectorType>
    MatrixType
    MultivariateGaussianModel<MatrixType, VectorType>::calculateExpectedFisherInformation(const VectorType &) const {
        return inverseCovariance;
    }

    template<typename MatrixType, typename VectorType>
    VectorType MultivariateGaussianModel<MatrixType, VectorType>::calculateGradient(const VectorType &x) const {
        return -inverseCovariance * (x - mean);
    }
}

#endif //HOPS_MULTIVARIATEGAUSSIANMODEL_HPP
