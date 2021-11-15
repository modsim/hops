#ifndef HOPS_SIMPLEXFACTORY_HPP
#define HOPS_SIMPLEXFACTORY_HPP

namespace hops {

    /**
     * @brief Factory for creating simplices.
     * @tparam Matrix
     * @tparam Vector
     */
    template<typename Matrix, typename Vector>
    class SimplexFactory {
    public:

        /**
         * @brief Creates \f$A\f$ and \f$b\f$ such that \f$Ax < b\f$ is a simplex.
         * @param numberOfDimensions
         * @return
         */
        static std::tuple<Matrix, Vector> createSimplex(long numberOfDimensions) {
            assert(numberOfDimensions > 0);
            Matrix A(numberOfDimensions + 1, numberOfDimensions);
            A << -Eigen::Matrix<typename Matrix::Scalar, Eigen::Dynamic, Eigen::Dynamic>::Identity(numberOfDimensions,
                                                                                                   numberOfDimensions),
                    Eigen::Matrix<typename Matrix::Scalar, Eigen::Dynamic, Eigen::Dynamic>::Ones(1, numberOfDimensions);
            Vector b(numberOfDimensions + 1);
            b << Eigen::Matrix<typename Matrix::Scalar, Eigen::Dynamic, 1>::Zero(numberOfDimensions),
                    1;
            return std::make_tuple(A, b);
        }
    };
}
#endif //HOPS_SIMPLEXFACTORY_HPP
