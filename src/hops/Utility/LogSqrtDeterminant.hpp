#ifndef HOPS_LOGSQRTDETERMINANT_HPP
#define HOPS_LOGSQRTDETERMINANT_HPP

namespace hops {
    template<typename MatrixSqrtType>
    double logSqrtDeterminant(const MatrixSqrtType &matrixSqrt) {
        return matrixSqrt.diagonal().array().log().sum();
    }
}

#endif //HOPS_LOGSQRTDETERMINANT_HPP
