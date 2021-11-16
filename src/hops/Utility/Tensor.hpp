#ifndef HOPS_4TENSOR_HPP
#define HOPS_4TENSOR_HPP

#include "VectorType.hpp"

namespace hops {
    template<size_t order>
    struct Tensor {
        VectorType data;
        long shape[order];
        long size = 0;

        Tensor(const VectorType& data, long shape[order]) :
                data(data), 
                shape(shape) {
            this->size = 1;
            for (size_t i = 0; i < order; ++i) {
                this->size *= shape[i];
            }
        }

        VectorType::Scalar& operator()(long index[order]) {
            long n = 0;
            long k = 1;
            for (size_t i = 0; i < order - 1; ++i) {
                k *= shape[i];
                n += index[i] * (size / k);
            }
            return data(n + index[order-1]);
        }
    };
}

#endif //HOPS_4TENSOR_HPP

