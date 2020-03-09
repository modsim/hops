#ifndef HOPS_POLYTOPESPACE_HPP
#define HOPS_POLYTOPESPACE_HPP

#include <utility>

namespace hops {
    // TODO maybe this class belongs to benchmark fixtures, because it should not be required anywhere else
    template<typename MatrixType, typename VectorType>
    struct PolytopeSpace {
        PolytopeSpace() = default;

        PolytopeSpace(MatrixType a, VectorType b, VectorType startingPoint) : A(std::move(a)),
                                                                              b(std::move(b)),
                                                                              startingPoint(std::move(startingPoint)) {}

        MatrixType A;
        VectorType b;
        VectorType startingPoint;
        MatrixType N;
        VectorType shift;

        MatrixType roundedA;
        VectorType roundedb;
        VectorType roundedStartingPoint;
        MatrixType roundedN;
        MatrixType roundedT;
        VectorType roundedShift;
    };
}

#endif //HOPS_POLYTOPESPACE_HPP
