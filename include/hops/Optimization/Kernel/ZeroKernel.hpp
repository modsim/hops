#ifndef HOPS_ZEROKERNEL_HPP
#define HOPS_ZEROKERNEL_HPP

#include <cmath>

namespace hops {
    template<typename MatrixType>
    class ZeroKernel {
    public:
        ZeroKernel () {
            // 
        }

        MatrixType operator()(const MatrixType& x, const MatrixType& y) {
            return MatrixType::Zero(x.size(), y.size());
        }
    };
}

#endif // HOPS_ZEROKERNEL_HPP
