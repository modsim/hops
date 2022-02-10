#ifndef HOPS_REFLECTOR_HPP
#define HOPS_REFLECTOR_HPP

#include <limits>

#include <hops/RandomNumberGenerator/RandomNumberGenerator.hpp>
#include <hops/Utility/VectorType.hpp>

namespace hops {

    /**
     * @brief Reflector-class helps reflecting states into polytope.
     */
    class Reflector {
    public:
        static constexpr double tolerance = 1e-15;
        /**
         * @brief For startPoint in polytope (Ax<b) the endPoint is reflected into the polytope. If the endPoint
         * is already in the endpoint it is returned.
         * @param A
         * @param b
         * @param startPoint
         * @param endPoint
         * @param epsilon numeric parameter. If a slack is smaller than epsilon, the trajectory is reflected at the relevant constraint
         * @paramt maxNumberOfReflections maximum number of reflections to compute before giving up
         * @return tuple of 1) boolean whether reflection was successful 2) number of reflections 3) reflected point when successful, otherwise endPoint
         */

        template<typename InternalMatrixType>
        static std::tuple<bool, long, VectorType> reflectIntoPolytope(const InternalMatrixType &A,
                                                                      const VectorType &b,
                                                                      const VectorType &startPoint,
                                                                      const VectorType &endPoint,
                                                                      long maxNumberOfReflections);
    };

    template<typename InternalMatrixType>
    std::tuple<bool, long, VectorType>
    Reflector::reflectIntoPolytope(const InternalMatrixType &A, const VectorType &b, const VectorType &startPoint,
                                   const VectorType &endPoint, long maxNumberOfReflections) {
        VectorType currentPoint = startPoint;
        VectorType trajectory = endPoint - startPoint;

        VectorType::Scalar trajectoryLength = trajectory.norm();
        VectorType::Scalar OriginalTrajectoryLength = trajectoryLength;
        VectorType trajectoryDirection = trajectory / trajectoryLength;

        // Used to implement kahan summation.
        double distanceTravelled = 0.;
        double distanceTravelledError = 0.;

        VectorType slacks = b - A * startPoint;

        Eigen::VectorXd activeConstraints = Eigen::VectorXd::Ones(slacks.rows());

        long numberOfReflections = 0;
        do {
            Eigen::VectorXd inverseDistancesToBorder = activeConstraints.cwiseProduct(
                    ((A * trajectoryDirection).cwiseQuotient(slacks)));

            double distanceToBorder = 1. / inverseDistancesToBorder.array().template unaryExpr(
                    [](double v) { return std::isfinite(v) ? v : -1; }).maxCoeff();

            if (distanceToBorder < 0) {
                distanceToBorder = std::numeric_limits<double>::max();
            }

            if (trajectoryLength < distanceToBorder) {
                currentPoint += trajectoryDirection * trajectoryLength;
                trajectoryLength = 0; // No remaining trajectoryLength to traverse
            } else {
                numberOfReflections++;
                double y = distanceToBorder - distanceTravelledError;
                double t = distanceTravelled + y;
                distanceTravelledError = (t-distanceTravelled) - y; // should be 0 if no rounding errors happen
                distanceTravelled = t;

                trajectoryLength = OriginalTrajectoryLength - distanceTravelled;
                currentPoint += trajectoryDirection * distanceToBorder;
                slacks.noalias() -= A * trajectoryDirection * distanceToBorder;
                for (int i = 0; i < A.rows(); ++i) {
                    if (slacks[i] <= tolerance) {
                        activeConstraints[i] = 0;
                        trajectoryDirection -= 2 * (trajectoryDirection.dot(A.row(i)) * A.row(i).transpose()) /
                                               A.row(i).squaredNorm();
                    } else {
                        activeConstraints[i] = 1;
                    }
                }
            }
        } while (trajectoryLength > 0 && numberOfReflections < maxNumberOfReflections);

        if (numberOfReflections < maxNumberOfReflections) {
            return std::make_tuple(true, numberOfReflections, currentPoint);
        }
        return std::make_tuple(false, numberOfReflections, endPoint);
    }
}

#endif //HOPS_REFLECTOR_HPP
