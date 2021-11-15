#ifndef HOPS_SQUAREDEXPONENTIALKERNEL_HPP
#define HOPS_SQUAREDEXPONENTIALKERNEL_HPP

#include <cmath>

namespace hops {
    template<typename MatrixType, typename VectorType>
    class SquaredExponentialKernel {
    public:
        SquaredExponentialKernel (double sigma = 1, double length = 1) :
                sigma(sigma),
                length(length) {
            // 
        }

        MatrixType operator()(const MatrixType& x, const MatrixType& y) {
            MatrixType covariance(x.rows(), y.rows());
            for (long i = 0; i < x.rows(); ++i) {
                for (long j = 0; j < y.rows(); ++j) {
                    VectorType diff = (x.row(i) - y.row(j)).transpose();
                    double squaredDistance = diff.transpose() * diff;
                    double val = sigma * sigma * std::exp(-0.5 * squaredDistance / (length * length));
                    covariance(i, j) = val;
                    //TODO
                    //covariance(j, i) = val;
                }
            }
            return covariance;
        }

        double sigma;
        double length;
    };
}

#endif // HOPS_SQUAREDEXPONENTIALKERNEL_HPP
