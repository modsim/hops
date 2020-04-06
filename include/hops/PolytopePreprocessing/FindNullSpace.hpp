#ifndef HOPS_FINDNULLSPACE_HPP
#define HOPS_FINDNULLSPACE_HPP

#include <Eigen/Core>

namespace hops {

    /**
     * @brief Finds the nullspace of S using SVD.
     * @tparam MatrixType
     * @tparam VectorType
     * @param S Stoichiometry and Constraints
     * @param b
     * @return
     */
    template<typename MatrixType, typename VectorType>
    std::tuple<Eigen::MatrixXd, Eigen::VectorXd>
    findNullSpace(const MatrixType& S, const VectorType& b) {

    };

}

#endif //HOPS_FINDNULLSPACE_HPP
