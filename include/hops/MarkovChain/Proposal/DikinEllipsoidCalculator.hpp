#ifndef HOPS_DIKINELLIPSOIDCALCULATOR_HPP
#define HOPS_DIKINELLIPSOIDCALCULATOR_HPP

#include <Eigen/Cholesky>
#include <Eigen/Core>
#include <Eigen/Dense>
#include <hops/FileWriter/CsvWriter.hpp>

namespace hops {
    template<typename MatrixType, typename VectorType>
    class DikinEllipsoidCalculator {
    public:
        DikinEllipsoidCalculator(MatrixType A, VectorType b);

        Eigen::Matrix<typename MatrixType::Scalar, Eigen::Dynamic, Eigen::Dynamic>
        calculateDikinEllipsoid(const VectorType &x) const;

    private:
        MatrixType A;
        VectorType b;
    };

    template<typename MatrixType, typename VectorType>
    DikinEllipsoidCalculator<MatrixType, VectorType>::DikinEllipsoidCalculator(MatrixType A, VectorType b) :
            A(std::move(A)), b(std::move(b)) {}

    template<typename MatrixType, typename VectorType>
    Eigen::Matrix<typename MatrixType::Scalar, Eigen::Dynamic, Eigen::Dynamic>
    DikinEllipsoidCalculator<MatrixType, VectorType>::calculateDikinEllipsoid(const VectorType &x) const {
        Eigen::Matrix<typename MatrixType::Scalar, Eigen::Dynamic, 1> inv_slack = (this->b -
                                                                                   this->A * x).cwiseInverse();

        Eigen::Matrix<typename MatrixType::Scalar, Eigen::Dynamic, Eigen::Dynamic> halfDikin =
                inv_slack.asDiagonal() * this->A;
        Eigen::Matrix<typename MatrixType::Scalar, Eigen::Dynamic, Eigen::Dynamic> dikin =
                halfDikin.transpose() * halfDikin;
        return dikin;
    }
}

#endif //HOPS_DIKINELLIPSOIDCALCULATOR_HPP
