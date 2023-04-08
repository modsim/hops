#define _USE_MATH_DEFINES

#include <cmath>
#include <Eigen/Dense>
#include <Eigen/LU>
#include <Eigen/SparseQR>
#include <iostream>

#include "hops/Polytope/MaximumVolumeEllipsoid.hpp"

template<typename Derived>
std::ostream &operator<<(std::ostream &out, const hops::MaximumVolumeEllipsoid<Derived> &maximumVolumeEllipsoid) {
    out << "Maximum Volume Ellipsoid:" << std::endl
        << maximumVolumeEllipsoid.maximumVolumeEllipsoid << std::endl
        << "center:" << std::endl
        << maximumVolumeEllipsoid.center << std::endl
        << "rounding transformation:" << std::endl
        << maximumVolumeEllipsoid.roundingTransformation << std::endl
        << "current error" << std::endl
        << maximumVolumeEllipsoid.currentError << std::endl;
    return out;
}

template<typename Derived>
void swap(hops::MaximumVolumeEllipsoid<Derived> &first, hops::MaximumVolumeEllipsoid<Derived> &second) {
    std::swap(first.roundingTransformation, second.roundingTransformation);
    std::swap(first.maximumVolumeEllipsoid, second.maximumVolumeEllipsoid);
    std::swap(first.center, second.center);
    std::swap(first.converged, second.converged);
    std::swap(first.iterations, second.iterations);
    std::swap(first.currentError, second.currentError);
    std::swap(first.tolerance, second.tolerance);
}

template<typename RealType>
Eigen::Matrix<RealType, Eigen::Dynamic, 1>
hops::MaximumVolumeEllipsoid<RealType>::applyRoundingTransformation(const Eigen::Matrix<RealType, Eigen::Dynamic, 1> &x) {
    return roundingTransformation.template triangularView<Eigen::Lower>() * x;
}

template<typename RealType>
RealType hops::MaximumVolumeEllipsoid<RealType>::computeVolume() const {
    const RealType halfDim = static_cast<RealType>(roundingTransformation.cols() * 0.5);
    return std::pow(M_PI, halfDim) / std::tgamma(halfDim + 1.0) * roundingTransformation.diagonal().prod();
}

template<typename RealType>
const Eigen::Matrix<RealType, Eigen::Dynamic, Eigen::Dynamic> &
hops::MaximumVolumeEllipsoid<RealType>::getRoundingTransformation() const {
    return roundingTransformation;
}

template<typename RealType>
const Eigen::Matrix<RealType, Eigen::Dynamic, 1> &hops::MaximumVolumeEllipsoid<RealType>::getCenter() const {
    return center;
}

template<typename RealType>
Eigen::Matrix<RealType, Eigen::Dynamic, Eigen::Dynamic> hops::MaximumVolumeEllipsoid<RealType>::getEllipsoid() const {
    return maximumVolumeEllipsoid;
}

template<typename RealType>
bool hops::MaximumVolumeEllipsoid<RealType>::hasConverged() const {
    return converged;
}

template<typename RealType>
size_t hops::MaximumVolumeEllipsoid<RealType>::getNumberOfIterations() const {
    return iterations;
}

template<typename RealType>
RealType hops::MaximumVolumeEllipsoid<RealType>::getCurrentError() const {
    return currentError;
}

template<typename RealType>
RealType hops::MaximumVolumeEllipsoid<RealType>::getTolerance() const {
    return tolerance;
}

template<typename RealType>
hops::MaximumVolumeEllipsoid<RealType>
hops::MaximumVolumeEllipsoid<RealType>::construct(const Eigen::Matrix<RealType, Eigen::Dynamic, Eigen::Dynamic> &Ain,
                                                  const Eigen::Matrix<RealType, Eigen::Dynamic, 1> &bin,
                                                  size_t maximumNumberOfIterationsToRun,
                                                  const Eigen::Matrix<RealType, Eigen::Dynamic, 1> &startingPoint,
                                                  RealType tolerance) {
    if (((Ain * startingPoint - bin).array() > std::numeric_limits<RealType>::epsilon()).any()) {
        std::cerr << "staring point is not in polytope " << std::endl;
        throw std::runtime_error("starting point is not in polytope");
    }
    const RealType minmu = 1e-8;
    const RealType tau0 = 0.75;

    const typename Eigen::Matrix<RealType, Eigen::Dynamic, Eigen::Dynamic>::Index m = Ain.rows();
    const typename Eigen::Matrix<RealType, Eigen::Dynamic, Eigen::Dynamic>::Index n = Ain.cols();
    const Eigen::Matrix<RealType, Eigen::Dynamic, 1> b = Eigen::Matrix<RealType, Eigen::Dynamic, 1>::Ones(m);
    const RealType bnrm = bin.norm();

    const long rank = Ain.colPivHouseholderQr().rank();
    if (rank < n) {
        throw std::runtime_error("Algorithm needs full column rank, because A^T * A has to be invertible");
    }

    const Eigen::Matrix<RealType, Eigen::Dynamic, 1> tempBmAx0 = bin - Ain * startingPoint;
    const Eigen::Matrix<RealType, Eigen::Dynamic, Eigen::Dynamic> bmAx0 = (tempBmAx0).cwiseInverse().asDiagonal();

    Eigen::SparseMatrix<RealType> A = (bmAx0 * Ain).sparseView();
    Eigen::Matrix<RealType, Eigen::Dynamic, 1> x = Eigen::Matrix<RealType, Eigen::Dynamic, 1>::Zero(n);
    Eigen::Matrix<RealType, Eigen::Dynamic, 1> y = Eigen::Matrix<RealType, Eigen::Dynamic, 1>::Ones(m);
    Eigen::Matrix<RealType, Eigen::Dynamic, 1> bmAx = b;

    RealType t = 0.0;
    Eigen::Matrix<RealType, Eigen::Dynamic, 1> z;
    Eigen::Matrix<RealType, Eigen::Dynamic, Eigen::Dynamic> E2;

    RealType res = 1.0;
    MaximumVolumeEllipsoid maximumVolumeEllipsoid;
    maximumVolumeEllipsoid.converged = false;
    maximumVolumeEllipsoid.iterations = 0;
    while (maximumVolumeEllipsoid.iterations < maximumNumberOfIterationsToRun) {
        maximumVolumeEllipsoid.iterations++;

        Eigen::Matrix<RealType, Eigen::Dynamic, Eigen::Dynamic> Y = y.asDiagonal();

        E2 = Eigen::Matrix<RealType, Eigen::Dynamic, Eigen::Dynamic>((A.transpose() * y.asDiagonal() * A)).inverse();
        Eigen::Matrix<RealType, Eigen::Dynamic, Eigen::Dynamic> Q = A * E2 * A.transpose();

        Eigen::Matrix<RealType, Eigen::Dynamic, 1> h = Q.diagonal().cwiseSqrt();

        if (maximumVolumeEllipsoid.iterations == 1) {
            t = (bmAx.cwiseQuotient(h)).minCoeff();
            const RealType t2 = t * t;
            y /= t2;
            h *= t;

            const Eigen::Matrix<RealType, Eigen::Dynamic, 1> temp = bmAx - h;
            z = temp.cwiseMax(1e-1);

            Q = t2 * Q;
            Y /= t2;
        }

        Eigen::Matrix<RealType, Eigen::Dynamic, 1> yz = y.cwiseProduct(z);
        Eigen::Matrix<RealType, Eigen::Dynamic, 1> yh = y.cwiseProduct(h);

        RealType gap = yz.sum() / m;
        RealType rmu = std::min(static_cast<RealType>(0.5), gap) * gap;
        rmu = std::max(rmu, minmu);

        Eigen::Matrix<RealType, Eigen::Dynamic, 1> R1 = -A.transpose() * yh;
        Eigen::Matrix<RealType, Eigen::Dynamic, 1> R2 = bmAx - h - z;

        Eigen::Matrix<RealType, Eigen::Dynamic, 1> R3 = rmu - yz.array();

        const RealType r1 = R1.template lpNorm<Eigen::Infinity>();
        const RealType r2 = R2.template lpNorm<Eigen::Infinity>();
        const RealType r3 = R3.template lpNorm<Eigen::Infinity>();
        res = std::max(r1, std::max(r2, r3));

        if ((res < tolerance * (1 + bnrm)) && (rmu <= minmu)) {
            maximumVolumeEllipsoid.converged = true;
            x += startingPoint;
            maximumVolumeEllipsoid.center = x;
            break;
        }

        Eigen::Matrix<RealType, Eigen::Dynamic, Eigen::Dynamic> YQ = Y * Q;
        Eigen::Matrix<RealType, Eigen::Dynamic, Eigen::Dynamic> YQQY = YQ.cwiseProduct(YQ.transpose());

        Eigen::Matrix<RealType, Eigen::Dynamic, 1> y2h = 2.0 * yh;
        Eigen::Matrix<RealType, Eigen::Dynamic, Eigen::Dynamic> YA = Y * A;

        Eigen::Matrix<RealType, Eigen::Dynamic, 1> tempProd = y2h.cwiseProduct(z);
        Eigen::Matrix<RealType, Eigen::Dynamic, 1> tempG = tempProd.cwiseMax(1e-12);
        Eigen::Matrix<RealType, Eigen::Dynamic, Eigen::Dynamic> tempG2 = tempG.asDiagonal();
        Eigen::Matrix<RealType, Eigen::Dynamic, Eigen::Dynamic> G = YQQY + tempG2;

        Eigen::Matrix<RealType, Eigen::Dynamic, 1> hz = h + z;
        Eigen::Matrix<RealType, Eigen::Dynamic, Eigen::Dynamic> tempHzYA = hz.asDiagonal() * YA;

        Eigen::Matrix<RealType, Eigen::Dynamic, Eigen::Dynamic> T = G.fullPivHouseholderQr().solve(tempHzYA);
        Eigen::Matrix<RealType, Eigen::Dynamic, Eigen::Dynamic> ATP = (y2h.asDiagonal() * T - YA).transpose();

        Eigen::Matrix<RealType, Eigen::Dynamic, 1> R3Dy = R3.cwiseQuotient(y);
        Eigen::Matrix<RealType, Eigen::Dynamic, 1> R23 = R2 - R3Dy;

        Eigen::Matrix<RealType, Eigen::Dynamic, 1> tempDxR = R1 + ATP * R23;
        Eigen::Matrix<RealType, Eigen::Dynamic, Eigen::Dynamic> tempDxL = ATP * A;
        Eigen::Matrix<RealType, Eigen::Dynamic, 1> dx = tempDxL.fullPivHouseholderQr().solve(tempDxR);

        Eigen::Matrix<RealType, Eigen::Dynamic, 1> Adx = A * dx;
        Eigen::Matrix<RealType, Eigen::Dynamic, 1> dyDy = G.fullPivHouseholderQr().solve(
                y2h.cwiseProduct(Adx - R23));

        Eigen::Matrix<RealType, Eigen::Dynamic, 1> dy = y.cwiseProduct(dyDy);
        Eigen::Matrix<RealType, Eigen::Dynamic, 1> dz = R3Dy - z.cwiseProduct(dyDy);

        RealType minA = 1.0;
        RealType tempMin = -0.5;
        RealType ax = -1.0 / std::min(tempMin, (-Adx.cwiseProduct(bmAx)).minCoeff());
        if (ax <= minA) {
            minA = ax;
        }
        RealType ay = -1.0 / std::min(tempMin, dyDy.minCoeff());
        if (ay <= minA) {
            minA = ay;
        }
        RealType az = -1.0 / std::min(tempMin, (dz.cwiseQuotient(z)).minCoeff());
        if (az <= minA) {
            minA = az;
        }
        RealType tau = std::max(tau0, 1 - res);
        RealType astep = tau * minA;

        x += astep * dx;
        y += astep * dy;
        z += astep * dz;

        bmAx -= astep * Adx;
    }

    Eigen::LLT<Eigen::Matrix<RealType, Eigen::Dynamic, Eigen::Dynamic>> chol(E2);
    maximumVolumeEllipsoid.roundingTransformation = chol.matrixL();
    maximumVolumeEllipsoid.maximumVolumeEllipsoid = E2;
    return maximumVolumeEllipsoid;
}

template<typename RealType>
hops::MaximumVolumeEllipsoid<RealType> hops::MaximumVolumeEllipsoid<RealType>::construct(
        const Eigen::Matrix<RealType, Eigen::Dynamic, Eigen::Dynamic> &A,
        const Eigen::Matrix<RealType, Eigen::Dynamic, 1> &b,
        size_t maximumNumberOfIterationsToRun,
        RealType tolerance) {
    std::unique_ptr<LinearProgram> linearProgram;
    linearProgram = LinearProgramFactory::createLinearProgram(A.template cast<double>(),
                                                              b.template cast<double>());
    Eigen::VectorXd startingPoint = linearProgram->computeChebyshevCenter().optimalParameters;
    return MaximumVolumeEllipsoid<RealType>::construct(A,
                                                       b,
                                                       maximumNumberOfIterationsToRun,
                                                       startingPoint.template cast<RealType>(),
                                                       tolerance);
}

template
class hops::MaximumVolumeEllipsoid<float>;

template
class hops::MaximumVolumeEllipsoid<double>;
