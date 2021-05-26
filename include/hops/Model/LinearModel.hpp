#ifndef HOPS_LINEARMODEL_HPP
#define HOPS_LINEARMODEL_HPP

#define _USE_MATH_DEFINES

#include <math.h>
#include <Eigen/Cholesky>
#include <Eigen/Core>
#include <Eigen/LU>
#include <utility>

namespace hops {
    template<typename Matrix, typename Vector>
    class LinearModel {
    public:
        using MatrixType = Matrix;
        using VectorType = Vector;

        LinearModel(VectorType measuredData, MatrixType dataCovariance, MatrixType linearModel);

        /**
         * @brief Evaluates the negative log likelihood for input x.
         * @param x
         * @return
         */
        typename MatrixType::Scalar calculateNegativeLogLikelihood(const VectorType &x) const;

        MatrixType calculateExpectedFisherInformation(const VectorType &) const;

        VectorType calculateLogLikelihoodGradient(const VectorType &x) const;

    private:
        VectorType measuredData;
        MatrixType dataCovariance;
        MatrixType inverseCovariance;
        MatrixType linearModel;
        typename MatrixType::Scalar logNormalizationConstant;
    };

    template<typename MatrixType, typename VectorType>
    LinearModel<MatrixType, VectorType>::LinearModel(
            VectorType measuredData,
            MatrixType dataCovariance,
            MatrixType linearModel) :
            measuredData(std::move(measuredData)),
            dataCovariance(std::move(dataCovariance)),
            linearModel(std::move(linearModel)) {
        Eigen::LLT<MatrixType, Eigen::Upper> solver(this->dataCovariance);
        Eigen::Matrix<typename MatrixType::Scalar, Eigen::Dynamic, Eigen::Dynamic> matrixL = solver.matrixL();
        Eigen::Matrix<typename MatrixType::Scalar, Eigen::Dynamic, Eigen::Dynamic> matrixU = solver.matrixU();
        Eigen::Matrix<typename MatrixType::Scalar, Eigen::Dynamic, Eigen::Dynamic> inverseMatrixL = matrixL.inverse();
        inverseCovariance = inverseMatrixL * inverseMatrixL.transpose();

        logNormalizationConstant = -static_cast<typename MatrixType::Scalar>(this->measuredData.rows()) / 2 *
                                   std::log(2 * M_PI)
                                   - matrixL.diagonal().array().log().sum();
    }

    template<typename MatrixType, typename VectorType>
    typename MatrixType::Scalar
    LinearModel<MatrixType, VectorType>::calculateNegativeLogLikelihood(const VectorType &x) const {
        return -logNormalizationConstant +
               0.5 * static_cast<typename MatrixType::Scalar>((linearModel * x - measuredData).transpose() *
                                                              inverseCovariance * (linearModel * x - measuredData));
    }

    template<typename MatrixType, typename VectorType>
    MatrixType
    LinearModel<MatrixType, VectorType>::calculateExpectedFisherInformation(const VectorType &) const {
        return inverseCovariance;
    }

    template<typename MatrixType, typename VectorType>
    VectorType
    LinearModel<MatrixType, VectorType>::calculateLogLikelihoodGradient(const VectorType &x) const {
        return -inverseCovariance * (linearModel*x - measuredData);
    }
}

#endif //HOPS_LINEARMODEL_HPP
