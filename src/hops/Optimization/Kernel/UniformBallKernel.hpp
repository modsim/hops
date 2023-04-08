#ifndef HOPS_UNIFORMBALLKERNEL_HPP
#define HOPS_UNIFORMBALLKERNEL_HPP

#include <cmath>

namespace hops {
    template<typename MatrixType, typename VectorType>
    class UniformBallKernel {
    public:
        UniformBallKernel (double length = 1) :
                length(length) {
            // 
        }

        MatrixType operator()(const MatrixType& x, const MatrixType& y) {
            MatrixType covariance(x.rows(), y.rows());
            for (long i = 0; i < x.rows(); ++i) {
                for (long j = 0; j < y.rows(); ++j) {
                    VectorType diff = (x.row(i) - y.row(j)).transpose();
                    double distance = std::sqrt(diff.transpose() * diff);
                    covariance(i, j) = static_cast<double>(distance <= length);
                }
            }
            return covariance;
        }

        double length;
    };
}

#endif // HOPS_UNIFORMBALLKERNEL_HPP
