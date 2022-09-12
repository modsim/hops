#ifndef HOPS_NORMALIZEPOLYTOPE_HPP
#define HOPS_NORMALIZEPOLYTOPE_HPP

#include <Eigen/Core>

namespace hops {

    /**
     * @brief Normalizes polytope defined by Ax < b
     * @deprecated Use PolyRound instead.
     * @tparam Derived1
     * @tparam Derived2
     * @param A Dense representation of A
     * @param b
     */
    template<typename Derived1, typename Derived2>
    void normalizePolytope(Eigen::MatrixBase<Derived1> &A, Eigen::MatrixBase<Derived2> &b) {
        for (int i = 0; i < A.rows(); ++i) {
            const double norm = A.row(i).template lpNorm<2>();
            A.row(i) /= norm;
            b(i) /= norm;
        }
    }
}


#endif //HOPS_NORMALIZEPOLYTOPE_HPP
