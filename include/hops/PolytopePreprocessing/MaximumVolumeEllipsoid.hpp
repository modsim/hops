#ifndef HOPS_MAXIMUMVOLUMEELLIPSOID_HPP
#define HOPS_MAXIMUMVOLUMEELLIPSOID_HPP

/**
 * @file MaxVolEllipsoid.hpp
 * @brief Based on an port from http://www.caam.rice.edu/~zhang/maxvep/ by Samuel Leweke
 *
 * @details
 *  Reference:
 *  Zhang, Y., & Gao, L. (2003):
 *  On Numerical Solution of the Maximum Volume Ellipsoid Problem.
 *  SIAM Journal on Optimization, 14(1), 53–76.
 *  doi:10.1137/S1052623401397230
 *
 *  Implementation initially ported from Matlab to C++ by Samuel Leweke (2013).
 */

#include <cmath>
#include <Eigen/Core>
#include <hops/LinearProgram/LinearProgramFactory.hpp>

namespace hops {
    template<typename RealType>
    class MaximumVolumeEllipsoid {
    public:
        MaximumVolumeEllipsoid(const MaximumVolumeEllipsoid &) = default;

        MaximumVolumeEllipsoid(MaximumVolumeEllipsoid &&) noexcept = default;

        MaximumVolumeEllipsoid &operator=(const MaximumVolumeEllipsoid &) = default;

        /**
         * @brief Applies the maximum volume ellipsoid to round vector.
         * @param x
         * @return
         */
        [[nodiscard]] Eigen::Matrix<RealType, Eigen::Dynamic, 1> applyRoundingTransformation(Eigen::Matrix<RealType, Eigen::Dynamic, 1> &x);

        [[nodiscard]] RealType calculateVolume() const;

        [[nodiscard]] const Eigen::Matrix<RealType, Eigen::Dynamic, Eigen::Dynamic> &getRoundingTransformation() const;

        [[nodiscard]] const Eigen::Matrix<RealType, Eigen::Dynamic, 1> &getCenter() const;

        [[nodiscard]] Eigen::Matrix<RealType, Eigen::Dynamic, Eigen::Dynamic> getEllipsoid() const;

        [[nodiscard]] bool hasConverged() const;

        [[nodiscard]] size_t getNumberOfIterations() const;

        [[nodiscard]] RealType getCurrentError() const;

        [[nodiscard]] RealType getTolerance() const;

        // TODO implement sparse construction for performance reasons
        static MaximumVolumeEllipsoid
        construct(const Eigen::Matrix<RealType, Eigen::Dynamic, Eigen::Dynamic> &A,
                  const Eigen::Matrix<RealType, Eigen::Dynamic, 1> &b,
                  size_t maximumNumberOfIterationsToRun,
                  const Eigen::Matrix<RealType, Eigen::Dynamic, 1> &startingPoint,
                  RealType tolerance = 1e-6);

        static MaximumVolumeEllipsoid
        construct(const Eigen::Matrix<RealType, Eigen::Dynamic, Eigen::Dynamic> &A,
                  const Eigen::Matrix<RealType, Eigen::Dynamic, 1> &b,
                  size_t maximumNumberOfIterationsToRun,
                  RealType tolerance = 1e-6);

        template<typename Derived>
        friend std::ostream &operator<<(std::ostream &out, const MaximumVolumeEllipsoid<Derived> &maximumVolumeEllipsoid);

    private:
        template<typename Derived>
        friend void swap(MaximumVolumeEllipsoid<Derived> &first, MaximumVolumeEllipsoid<Derived> &second);

        MaximumVolumeEllipsoid() = default;

        Eigen::Matrix<RealType, Eigen::Dynamic, Eigen::Dynamic> roundingTransformation;
        Eigen::Matrix<RealType, Eigen::Dynamic, Eigen::Dynamic> maximumVolumeEllipsoid;
        Eigen::Matrix<RealType, Eigen::Dynamic, 1> center;

        size_t iterations = 0;
        RealType currentError = 0;
        RealType tolerance = 0;
        bool converged = false;
    };
}

#endif //HOPS_MAXIMUMVOLUMEELLIPSOID_HPP