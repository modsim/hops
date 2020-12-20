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
        auto inv_slack = (this->b - this->A * x).array().pow(2).inverse().matrix().asDiagonal();
        return this->A.transpose() * inv_slack * A;
    }
}

#endif //HOPS_DIKINELLIPSOIDCALCULATOR_HPP
