#ifndef HOPS_REFLECTOR_HPP
#define HOPS_REFLECTOR_HPP

#include <limits>

#include "hops/RandomNumberGenerator/RandomNumberGenerator.hpp"
#include "hops/Utility/VectorType.hpp"

namespace hops {

    /**
     * @brief Reflector-class helps reflecting states into polytope.
     */
    class Reflector {
    public:
        static constexpr double tolerance = 1e-15;

        /**
         * @brief For startPoint in polytope (Ax<inequalityLhs) the endPoint is reflected into the polytope. If the endPoint
         * is already in the endpoint it is returned.
         * @param inequalityConstraintMatrix
         * @param inequalityLhs
         * @param startPoint
         * @param endPoint
         * @param epsilon numeric parameter. If a slack is smaller than epsilon, the trajectory is reflected at the relevant constraint
         * @paramt maxReflections maximum number of reflections to compute before giving up
         * @return tuple of 1) boolean whether reflection was successful 2) number of reflections 3) reflected point when successful, otherwise endPoint
         */
        template<typename InternalMatrixType>
        static std::tuple<bool, long, VectorType>
        reflectIntoPolytope(const InternalMatrixType &inequalityConstraintMatrix,
                            const VectorType &inequalityLhs,
                            const VectorType &startPoint,
                            const VectorType &endPoint,
                            long maxNumberOfReflections);

        /**
         * @brief For startPoint in polytope (Ax<inequalityLhs) the endPoint is reflected into the polytope. If the endPoint
         * is already in the endpoint it is returned.
         * @param inequalityConstraintMatrix
         * @param inequalityLhs
         * @param quadraticConstraintMatrix
         * @param quadraticConstraintOffset
         * @param quadraticConstraintLhs
         * @param startPoint
         * @param endPoint
         * @param epsilon numeric parameter. If a slack is smaller than epsilon, the trajectory is reflected at the relevant constraint
         * @paramt maxReflections maximum number of reflections to compute before giving up
         * @return tuple of 1) boolean whether reflection was successful 2) number of reflections 3) reflected point when successful, otherwise endPoint
         */
        template<typename InternalMatrixType>
        static std::tuple<bool, long, VectorType>
        reflectIntoPolytope(const InternalMatrixType &inequalityConstraintMatrix,
                            const VectorType &inequalityLhs,
                            const InternalMatrixType &quadraticConstraintMatrix,
                            const VectorType &quadraticConstraintOffset,
                            double quadraticConstraintLhs,
                            const VectorType &startPoint,
                            const VectorType &endPoint,
                            long maxNumberOfReflections);
    };

    template<typename InternalMatrixType>
    std::tuple<bool, long, VectorType>
    Reflector::reflectIntoPolytope(const InternalMatrixType &inequalityConstraintMatrix,
                                   const VectorType &inequalityLhs,
                                   const VectorType &startPoint,
                                   const VectorType &endPoint,
                                   long maxNumberOfReflections) {
        VectorType currentPoint = startPoint;
        VectorType trajectory = endPoint - startPoint;

        VectorType::Scalar trajectoryLength = trajectory.norm();
        VectorType::Scalar OriginalTrajectoryLength = trajectoryLength;
        VectorType trajectoryDirection = trajectory / trajectoryLength;

        // Used to implement kahan summation.
        double distanceTravelled = 0.;
        double distanceTravelledError = 0.;

        VectorType slacks = inequalityLhs - inequalityConstraintMatrix * startPoint;

        Eigen::VectorXd activeConstraints = Eigen::VectorXd::Ones(slacks.rows());

        long numberOfReflections = 0;
        do {
            Eigen::VectorXd inverseDistancesToBorder = activeConstraints.cwiseProduct(
                    ((inequalityConstraintMatrix * trajectoryDirection).cwiseQuotient(slacks)));

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
                distanceTravelledError = (t - distanceTravelled) - y; // should be 0 if no rounding errors happen
                distanceTravelled = t;

                trajectoryLength = OriginalTrajectoryLength - distanceTravelled;
                currentPoint += trajectoryDirection * distanceToBorder;
                slacks.noalias() -= inequalityConstraintMatrix * trajectoryDirection * distanceToBorder;
                for (int i = 0; i < inequalityConstraintMatrix.rows(); ++i) {
                    if (slacks[i] <= tolerance) {
                        activeConstraints[i] = 0;
                        trajectoryDirection -= 2 * (trajectoryDirection.dot(inequalityConstraintMatrix.row(i)) *
                                                    inequalityConstraintMatrix.row(i).transpose()) /
                                               inequalityConstraintMatrix.row(i).squaredNorm();
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

    template<typename InternalMatrixType>
    std::tuple<bool, long, VectorType>
    Reflector::reflectIntoPolytope(const InternalMatrixType &inequalityConstraintMatrix,
                                   const VectorType &inequalityLhs,
                                   const InternalMatrixType &quadraticConstraintMatrix,
                                   const VectorType &quadraticConstraintOffset,
                                   double quadraticConstraintLhs,
                                   const VectorType &startPoint,
                                   const VectorType &endPoint,
                                   long maxNumberOfReflections) {
        VectorType currentPoint = startPoint;
        VectorType trajectory = endPoint - startPoint;

        VectorType::Scalar trajectoryLength = trajectory.norm();
        VectorType::Scalar OriginalTrajectoryLength = trajectoryLength;
        VectorType trajectoryDirection = trajectory / trajectoryLength;

        // Used to implement kahan summation.
        double distanceTravelled = 0.;
        double distanceTravelledError = 0.;

        VectorType slacks = inequalityLhs - inequalityConstraintMatrix * startPoint;

        hops::VectorType activeConstraints = Eigen::VectorXd::Ones(slacks.rows());
        InternalMatrixType squaredQuadraticConstraintMatrix =
                quadraticConstraintMatrix.transpose() * quadraticConstraintMatrix;

        long numberOfReflections = 0;
        do {
            Eigen::VectorXd inverseDistancesToBorder = activeConstraints.cwiseProduct(
                    ((inequalityConstraintMatrix * trajectoryDirection).cwiseQuotient(slacks)));

            double distanceToLinearConstraints = 1. / inverseDistancesToBorder.array().template unaryExpr(
                    [](double v) { return std::isfinite(v) ? v : -1; }).maxCoeff();

            // set up p-q-formula for ellispoid distance
            double quadraticNorm = trajectoryDirection.transpose() * quadraticConstraintMatrix * trajectoryDirection;
            VectorType directionTranspose = trajectoryDirection.transpose();

            double scaling = 2 / quadraticNorm;
            double _p = (trajectoryDirection.transpose() *
                         quadraticConstraintMatrix *
                         (currentPoint - quadraticConstraintOffset));
            double p = scaling * _p;

            double q = ((currentPoint - quadraticConstraintOffset).transpose() *
                        quadraticConstraintMatrix *
                        (currentPoint - quadraticConstraintOffset) - quadraticConstraintLhs)
                       / quadraticNorm;

            double forwardDistanceToQuadraticConstraints = -p / 2 + std::sqrt(std::pow(p / 2, 2) - q);
            // backwards distance not required
            // double backWardsDistanceToQuadraticConstraints = -p/2 - std::sqrt(std::pow(p/2, 2) - q);
            double distanceToQuadraticConstraints = forwardDistanceToQuadraticConstraints;

            double distanceToBorder = std::min(distanceToLinearConstraints, distanceToQuadraticConstraints);

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
                distanceTravelledError = (t - distanceTravelled) - y; // should be 0 if no rounding errors happen
                distanceTravelled = t;

                trajectoryLength = OriginalTrajectoryLength - distanceTravelled;
                currentPoint += trajectoryDirection * distanceToBorder;
                slacks.noalias() -= inequalityConstraintMatrix * trajectoryDirection * distanceToBorder;
                for (int i = 0; i < inequalityConstraintMatrix.rows(); ++i) {
                    if (distanceToLinearConstraints < distanceToQuadraticConstraints) {
                        // reflect on linear constraints
                        if (slacks[i] <= tolerance) {
                            activeConstraints[i] = 0;
                            trajectoryDirection -= 2 * (trajectoryDirection.dot(inequalityConstraintMatrix.row(i)) *
                                                        inequalityConstraintMatrix.row(i).transpose()) /
                                                   inequalityConstraintMatrix.row(i).squaredNorm();
                        } else {
                            activeConstraints[i] = 1;
                        }
                    } else {
                        // Reflect on quadratic constraints (ellipsoid), which always hits in right angle
                        trajectoryDirection = -trajectoryDirection;
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
