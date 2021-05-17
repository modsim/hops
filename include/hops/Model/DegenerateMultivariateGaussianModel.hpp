#ifndef HOPS_DEGENERATEMULTIVARIATEGAUSSIANMODEL_HPP
#define HOPS_DEGENERATEMULTIVARIATEGAUSSIANMODEL_HPP

#define _USE_MATH_DEFINES
#include <math.h>
#include <Eigen/Cholesky>
#include <Eigen/Core>
#include <Eigen/LU>
#include <utility>

namespace hops {
    template<typename Matrix, typename Vector>
    class DegenerateMultivariateGaussianModel {
    public:
        using MatrixType = Matrix;
        using VectorType = Vector;

        DegenerateMultivariateGaussianModel(VectorType mean, MatrixType covariance, std::vector<long> inactive = std::vector<long>(0));

        typename MatrixType::Scalar calculateNegativeLogLikelihood(const VectorType &x) const;

        MatrixType calculateExpectedFisherInformation(const VectorType &) const;

        VectorType calculateLogLikelihoodGradient(const VectorType &x) const;

    private:
        VectorType mean;
        MatrixType covariance;
        std::vector<long> inactive;
        MatrixType inverseCovariance;
        typename MatrixType::Scalar logNormalizationConstant;

        void removeRow(Eigen::MatrixXd& matrix, unsigned int rowToRemove) const {
            unsigned int numRows = matrix.rows()-1;
            unsigned int numCols = matrix.cols();

            if (rowToRemove < numRows) {
                matrix.block(rowToRemove,0,numRows-rowToRemove,numCols) = matrix.bottomRows(numRows-rowToRemove);
            }

            matrix.conservativeResize(numRows,numCols);
        }

        void removeColumn(Eigen::MatrixXd& matrix, unsigned int colToRemove) const {
            unsigned int numRows = matrix.rows();
            unsigned int numCols = matrix.cols()-1;

            if (colToRemove < numCols) {
                matrix.block(0,colToRemove,numRows,numCols-colToRemove) = matrix.rightCols(numCols-colToRemove);
            }

            matrix.conservativeResize(numRows,numCols);
        }

        void removeRow(Eigen::VectorXd& vector, unsigned int rowToRemove) const {
            unsigned int numRows = vector.rows()-1;

            if (rowToRemove < numRows) {
                vector.segment(rowToRemove,numRows-rowToRemove) = vector.tail(numRows-rowToRemove);
            }

            vector.conservativeResize(numRows);
        }

        void stripInactive (Eigen::MatrixXd& matrix) const {
            for (auto& i : inactive) {
                removeRow(matrix, i);
                removeColumn(matrix, i);
            }
        }

        void stripInactive (Eigen::VectorXd& vector) const {
            for (auto& i : inactive) {
                removeRow(vector, i);
            }
        }
    };

    template<typename MatrixType, typename VectorType>
    DegenerateMultivariateGaussianModel<MatrixType, VectorType>::DegenerateMultivariateGaussianModel(VectorType mean,
                                                                                 MatrixType covariance,
                                                                                 std::vector<long> inactive) :
        mean(mean),
        covariance(covariance),
        inactive(inactive)
    {
        stripInactive(this->mean);
        stripInactive(this->covariance);

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
    DegenerateMultivariateGaussianModel<MatrixType, VectorType>::calculateNegativeLogLikelihood(const VectorType &x) const {
        VectorType _x = x;
        stripInactive(_x);
        return -logNormalizationConstant +
               0.5 * static_cast<typename MatrixType::Scalar>((_x - mean).transpose() * inverseCovariance * (_x - mean));
    }

    template<typename MatrixType, typename VectorType>
    MatrixType
    DegenerateMultivariateGaussianModel<MatrixType, VectorType>::calculateExpectedFisherInformation(const VectorType &) const {
        return inverseCovariance;
    }

    template<typename MatrixType, typename VectorType>
    VectorType
    DegenerateMultivariateGaussianModel<MatrixType, VectorType>::calculateLogLikelihoodGradient(const VectorType &x) const {
        VectorType _x = x;
        stripInactive(_x);
        return -inverseCovariance * (_x - mean);
    }
}

#endif //HOPS_DEGENERATEMULTIVARIATEGAUSSIANMODEL_HPP
