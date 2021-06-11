#ifndef HOPS_GAUSSIANPROCESS_HPP
#define HOPS_GAUSSIANPROCESS_HPP

#include "../RandomNumberGenerator/RandomNumberGenerator.hpp"

#include <Eigen/Core>
#include <Eigen/LU>
#include <Eigen/SVD>

#include <random>
#include <vector>
#include <cmath>

#include <iostream>

namespace hops {
    template<typename MatrixType, typename VectorType, typename Kernel>
    class GaussianProcess {
    public:
        GaussianProcess (Kernel kernel, double constantPriorMean = 0) :
                kernel(kernel) {
            priorMeanFunction = [=](VectorType) -> double { return constantPriorMean; };
        }

        GaussianProcess (Kernel kernel, std::function<double (VectorType)> priorMeanFunction) :
                priorMeanFunction(priorMeanFunction),
                kernel(kernel) {
            //
        }

        GaussianProcess getPriorCopy() {
            return GaussianProcess<MatrixType, VectorType, Kernel>(this->kernel, this->priorMeanFunction);
        }

        GaussianProcess getPosteriorCopy() {
            GaussianProcess<MatrixType, VectorType, Kernel> gp = this->getPriorCopy();
            gp.sampleInputs = sampleInputs;
            gp.posteriorMean = posteriorMean;
            gp.posteriorCovariance = posteriorCovariance;
            gp.sqrtInvPosteriorCovariance = sqrtInvPosteriorCovariance;
            gp.observedCovariance = observedCovariance;
            gp.invObservedCovariance = invObservedCovariance;
            gp.observedInputs = observedInputs;
            gp.observedValues = observedValues;
            gp.observedValueErrors = observedValueErrors;
            return gp;
        }

        std::vector<double> sample(const std::vector<VectorType>& x, 
                                   hops::RandomNumberGenerator& randomNumberGenerator) {
            size_t max;
            return sample(x, randomNumberGenerator, max);
        }

        /**
         * Sample from posterior at x
         *
         */
        std::vector<double> sample(const std::vector<VectorType>& x, 
                                   hops::RandomNumberGenerator& randomNumberGenerator, 
                                   size_t& maxElement) {
            MatrixType Ks = kernel(observedInputs, x);
            MatrixType Kss = kernel(x, x);

            posteriorMean = priorMean(x) + Ks.transpose() * invObservedCovariance * (observedValues - priorMean(observedInputs));
            posteriorCovariance = Kss - Ks.transpose() * invObservedCovariance * Ks;

            Eigen::BDCSVD<MatrixType> solver(MatrixType(posteriorCovariance), Eigen::ComputeFullU);
            sqrtInvPosteriorCovariance = solver.matrixU() * 
                                                     solver.singularValues().cwiseSqrt().asDiagonal();

            VectorType drawVec(x.size());
            auto standardNormal = std::normal_distribution<double>();

            for (size_t i = 0; i < x.size(); ++i) {
                drawVec(i) = standardNormal(randomNumberGenerator);
                assert(!std::isnan(drawVec(i)));
            }

            //std::cout << sqrtInvPosteriorCovariance << std::endl;
            drawVec = posteriorMean + sqrtInvPosteriorCovariance * drawVec;

            maxElement = 0;
            std::vector<double> draw(x.size());
            for (size_t i = 0; i < x.size(); ++i) {
                assert(!std::isnan(drawVec(i)));
                draw[i] = drawVec(i);
                if (draw[i] > draw[maxElement]) {
                    maxElement = i;
                }
            }

            return draw;
        }

        void addObservations(const std::vector<VectorType>& x, const std::vector<double>& y, const std::vector<double>& error) {
            assert(x.size() == y.size());
            assert(y.size() == error.size());

            VectorType newObservedValues = VectorType(observedValues.rows() + y.size());
            newObservedValues.head(observedValues.rows()) = observedValues;

            VectorType newObservedValueErrors = VectorType(observedValueErrors.rows() + error.size());
            newObservedValueErrors.head(observedValueErrors.rows()) = observedValueErrors;

            for (size_t i = 0; i < x.size(); ++i) {
                observedInputs.push_back(x[i]);
                newObservedValues(observedValues.rows() + i) = y[i];
                newObservedValueErrors(observedValueErrors.rows() + i) = error[i];
            }

            observedValues = newObservedValues;
            observedValueErrors = newObservedValueErrors;

            observedCovariance = kernel(observedInputs, observedInputs);
            observedCovariance = observedCovariance + Eigen::MatrixXd(observedValueErrors.asDiagonal());

            if (observedCovariance.size() > 0) {
                invObservedCovariance = observedCovariance.inverse();
            } else {
                invObservedCovariance = Eigen::MatrixXd::Zero(0, 0);
            }
        }

        void addObservations(const std::vector<VectorType>& x, const std::vector<double>& y) {
            addObservations(x, y, std::vector<double>(y.size(), 0));
        }

        void addObservation(const VectorType& x, double y, double error = 0) {
            observedInputs.push_back(x);

            VectorType newObservedValues = VectorType(observedValues.rows() + 1);
            newObservedValues.head(observedValues.rows()) = observedValues;
            newObservedValues(observedValues.rows()) = y;
            observedValues = newObservedValues;

            VectorType newObservedValueErrors = VectorType(observedValueErrors.rows() + 1);
            newObservedValueErrors.head(observedValueErrors.rows()) = observedValueErrors;
            newObservedValueErrors(observedValueErrors.rows()) = error;
            observedValueErrors = newObservedValueErrors;

            observedCovariance = kernel(observedInputs, observedInputs);
            observedCovariance += observedValueErrors.asDiagonal();

            if (observedCovariance.size() > 0) {
                invObservedCovariance = observedCovariance.inverse();
            } else {
                invObservedCovariance = Eigen::MatrixXd::Zero(0, 0);
            }
        }

        const VectorType& getPosteriorMean() const { return posteriorMean; }
        const MatrixType& getPosteriorCovariance() const { return posteriorCovariance; }
        const MatrixType& getSqrtInvPosteriorCovariance() const { return sqrtInvPosteriorCovariance; }

    private:
        std::function<double (VectorType)> priorMeanFunction;
        Kernel kernel;

        std::vector<VectorType> sampleInputs;
        VectorType posteriorMean;
        MatrixType posteriorCovariance;
        MatrixType sqrtInvPosteriorCovariance;

        MatrixType observedCovariance;
        MatrixType invObservedCovariance;
        
        std::vector<VectorType> observedInputs;
        VectorType observedValues;
        VectorType observedValueErrors;

        VectorType priorMean(const std::vector<VectorType>& x) {
            VectorType prior(x.size());
            for (size_t i = 0; i < x.size(); ++i) {
                prior(i) = priorMeanFunction(x[i]);
            }
            return prior;
        }
    };

    template<typename MatrixType, typename VectorType>
    class SquaredExponentialKernel {
    public:
        SquaredExponentialKernel (double sigma = 1, double length = 1) :
                sigma(sigma),
                length(length) {
            // 
        }

        MatrixType operator()(std::vector<VectorType> x, std::vector<VectorType> y) {
            MatrixType covariance(x.size(), y.size());
            for (size_t i = 0; i < x.size(); ++i) {
                for (size_t j = 0; j < y.size(); ++j) {
                    VectorType diff = x[i] - y[j];
                    double squaredDistance = diff.transpose() * diff;
                    covariance(i, j) = sigma * sigma * std::exp(-0.5 * squaredDistance / (length * length));
                }
            }
            return covariance;
        }
    private:
        double sigma;
        double length;
    };
}

#endif // HOPS_GAUSSIANPROCESS_HPP
