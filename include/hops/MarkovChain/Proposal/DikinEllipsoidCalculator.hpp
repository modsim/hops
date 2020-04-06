#ifndef HOPS_DIKINELLIPSOIDCALCULATOR_HPP
#define HOPS_DIKINELLIPSOIDCALCULATOR_HPP

#include <Eigen/Cholesky>
#include <Eigen/Core>

namespace hops {
    template<typename MatrixType, typename VectorType>
    class DikinEllipsoidCalculator {
    public:
        DikinEllipsoidCalculator(MatrixType A, VectorType b);

        Eigen::Matrix<typename MatrixType::Scalar, Eigen::Dynamic, Eigen::Dynamic>
        calculateCholeskyFactorOfDikinEllipsoid(const VectorType &x);

        Eigen::Matrix<typename MatrixType::Scalar, Eigen::Dynamic, Eigen::Dynamic>
        calculateDikinEllipsoid(const VectorType &x);

    private:
        MatrixType A;
        VectorType b;
    };

    template<typename MatrixType, typename VectorType>
    DikinEllipsoidCalculator<MatrixType, VectorType>::DikinEllipsoidCalculator(MatrixType A, VectorType b) :
            A(std::move(A)), b(std::move(b)) {}

    template<typename MatrixType, typename VectorType>
    Eigen::Matrix<typename MatrixType::Scalar, Eigen::Dynamic, Eigen::Dynamic>
    DikinEllipsoidCalculator<MatrixType, VectorType>::calculateCholeskyFactorOfDikinEllipsoid(const VectorType &x) {
        Eigen::Matrix<typename MatrixType::Scalar, Eigen::Dynamic, Eigen::Dynamic> dikinEllipsoid =
                calculateDikinEllipsoid(x);

        Eigen::LLT<Eigen::Matrix<typename MatrixType::Scalar, Eigen::Dynamic, Eigen::Dynamic>> solver(dikinEllipsoid);
        if (solver.info() != Eigen::Success) {
            Eigen::LDLT<Eigen::Matrix<typename MatrixType::Scalar, Eigen::Dynamic, Eigen::Dynamic>> semiPositiveDefiniteCholeskySolver(
                    dikinEllipsoid);
            if (semiPositiveDefiniteCholeskySolver.info() != Eigen::Success) {
                throw std::runtime_error(
                        std::string("Error in cholesky factorization of dikin ellipsoid. Solver status: ") +
                        std::to_string(solver.info())
                );
            }
            return semiPositiveDefiniteCholeskySolver.vectorD().cwiseAbs().cwiseSqrt().asDiagonal() *
                   Eigen::MatrixXd(semiPositiveDefiniteCholeskySolver.matrixL());
        } else {
            return solver.matrixL();
        }
    }

    template<typename MatrixType, typename VectorType>
    Eigen::Matrix<typename MatrixType::Scalar, Eigen::Dynamic, Eigen::Dynamic>
    DikinEllipsoidCalculator<MatrixType, VectorType>::calculateDikinEllipsoid(const VectorType &x) {
        return A.transpose() * ((b - A * x).array().pow(2).inverse().matrix().asDiagonal()) * A;
    }
}

#endif //HOPS_DIKINELLIPSOIDCALCULATOR_HPP
