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
                kernel(kernel), isNewObservations(true) {
            priorMeanFunction = [=](VectorType) -> double { return constantPriorMean; };
        }

        GaussianProcess (Kernel kernel, std::function<double (VectorType)> priorMeanFunction) :
                priorMeanFunction(priorMeanFunction),
                kernel(kernel),
                isNewObservations(true) {
            //
        }

        GaussianProcess getPriorCopy() {
            return GaussianProcess<MatrixType, VectorType, Kernel>(this->kernel, this->priorMeanFunction);
        }

        GaussianProcess getPosteriorCopy() {
            GaussianProcess<MatrixType, VectorType, Kernel> gp = this->getPriorCopy();
            gp.storedInputs = storedInputs;
            gp.inputPriorMean = inputPriorMean;
            //gp.inputAggregatedErrors = inputAggregatedErrors;
            gp.observationInputCovariance = observationInputCovariance;
            gp.inputCovariance = inputCovariance;

            gp.posteriorMean = posteriorMean;
            gp.posteriorCovariance = posteriorCovariance;
            gp.sqrtPosteriorCovariance = sqrtPosteriorCovariance;

            gp.isNewObservations = isNewObservations;
            gp.observedCovariance = observedCovariance;
            gp.sqrtObservedCovariance = sqrtObservedCovariance;
            
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
         *  Compute the posterior mean and covariance for the given input. This method checks, if any data has changed before 
         *  recomputing depending quantities, which makes it rather safe to call it before sampling or querying any posterior data
         *
         */
        void computePosterior(const std::vector<VectorType>& input) {
            bool isNewInput = input != storedInputs;
            if (isNewObservations || isNewInput) {
                observationInputCovariance = kernel(observedInputs, input);
                //inputAggregatedErrors = aggregateErrors(input);
                
                if (isNewInput) {
                    inputCovariance = kernel(input, input);
                    inputPriorMean = priorMean(input);

                    storedInputs = input;
                }

                if (isNewObservations) {
                    posteriorMean = sqrtObservedCovariance.template triangularView<Eigen::Lower>().solve(observedValues - priorMean(observedInputs));
                    posteriorMean = sqrtObservedCovariance.template triangularView<Eigen::Lower>().transpose().solve(posteriorMean);

                    posteriorCovariance = sqrtObservedCovariance.template triangularView<Eigen::Lower>().solve(observationInputCovariance);
                    posteriorCovariance = sqrtObservedCovariance.template triangularView<Eigen::Lower>().transpose().solve(posteriorCovariance);
                }

//#ifndef NDEBUG
//            MatrixType invObservedCovariance = observedCovariance.inverse();
//            MatrixType control = invObservedCovariance * (observedValues - priorMean(observedInputs));
//            control -= posteriorMean;
//            assert((control.size() == 0 || control.cwiseAbs().maxCoeff() < 1.e-5) && "computing the inverse for control might have failed.");
//#endif 

                posteriorMean = inputPriorMean + observationInputCovariance.transpose() * posteriorMean;

//#ifndef NDEBUG
//            control = invObservedCovariance * observationInputCovariance;
//            control -= posteriorCovariance;
//            assert((control.size() == 0 || control.cwiseAbs().maxCoeff() < 1.e-5) && "computing the inverse for control might have failed.");
//#endif 

                posteriorCovariance = inputCovariance - observationInputCovariance.transpose() * posteriorCovariance;
                //posteriorCovariance += inputAggregatedErrors.asDiagonal();

                Eigen::BDCSVD<MatrixType> solver(MatrixType(posteriorCovariance), Eigen::ComputeFullU);
                sqrtPosteriorCovariance = solver.matrixU() * solver.singularValues().cwiseSqrt().asDiagonal();

                isNewObservations = false;
            }
        }

        /**
         * Sample from posterior at x
         *
         */
        Eigen::VectorXd sample(const std::vector<VectorType>& input, 
                               hops::RandomNumberGenerator& randomNumberGenerator, 
                               size_t& maxElement) {
            computePosterior(input); 

            VectorType draw(input.size());
            auto standardNormal = std::normal_distribution<double>();

            for (size_t i = 0; i < input.size(); ++i) {
                draw(i) = standardNormal(randomNumberGenerator);
                assert(!std::isnan(draw(i)));
            }

            draw = posteriorMean + sqrtPosteriorCovariance * draw;
            draw.maxCoeff(&maxElement);

            return draw;
        }

        Eigen::VectorXd sample(hops::RandomNumberGenerator& randomNumberGenerator, 
                               size_t& maxElement) {
            return sample(storedInputs, randomNumberGenerator, maxElement);
        }

        /**
         *  Given an observation (x_i, y_i, eps_i) stored in *this, if there is an x_j in the argument x passed, s.t. x_i = x_j, then
         *  y_i := x_j and eps_i := y_j
         *
         *  All x_j and the respective y_j and eps_j from the passed arguments, which were not found in *this, will be stored in the reference arguments
         *  x, y and error.
         *
         *  The isUnique argument controls, whether x may be assumed to be unique in *this. If isUnique == true, then only the data at the first 
         *  occurence of x_i in *this will be updated.
         */
        void updateObservations(std::vector<VectorType>& x, std::vector<double>& y, std::vector<double>& error, bool isUnique = false) {
            assert(x.size() == y.size());
            assert(y.size() == error.size());

            std::vector<VectorType> inputsNotFound;
            std::vector<double> valuesNotFound;
            std::vector<double> errorsNotFound;

            // collect indices of observations which should be updated
            std::vector<size_t> updateCandidates;
            for (size_t i = 0; i < x.size(); ++i) {
                bool foundInput = false;
                for (size_t j = 0; j < observedInputs.size(); ++j) {
                    if (x[i] == observedInputs[j]) {
                        updateCandidates.push_back(j);
                        foundInput = true;

                        if (isUnique) {
                            break;
                        }
                    }
                }

                // record data which could not be updated, because it was not stored in *this the first place,
                // in order to add it afterwards
                if (!foundInput) {
                    inputsNotFound.push_back(x[i]);
                    valuesNotFound.push_back(y[i]);
                    errorsNotFound.push_back(error[i]);
                }
            }

            // if any entries were updated, set the isNewObservations flag
            if (updateCandidates.size() > 0) {
                isNewObservations = true;
            }

            for (size_t i = 0; i < updateCandidates.size(); ++i) {
                size_t k = updateCandidates[i];

                // update the data
                observedValues(k) = y[i];
                observedValueErrors(k) = error[i];

                // update the observed covariance 
                observedCovariance.row(k) = kernel({observedInputs[k]}, observedInputs);
                observedCovariance.col(k) = observedCovariance.row(k).transpose();
                observedCovariance(k, k) += observedValueErrors(k);
                //observedCovariance(k, k) += 1.e-5;
            }

//#ifndef NDEBUG
//            MatrixType control = kernel(observedInputs, observedInputs);
//            control -= observedCovariance;
//            assert(control.size() == 0 || control.cwiseAbs().maxCoeff() < 1.e-5);
//#endif

            // compute the cholesky factorization on the new observed covariance
            sqrtObservedCovariance = observedCovariance.llt().matrixL();
        
            x = inputsNotFound;
            y = valuesNotFound;
            error = errorsNotFound;
        }

        void addObservations(const std::vector<VectorType>& x, const std::vector<double>& y, const std::vector<double>& error) {
            assert(x.size() == y.size());
            assert(y.size() == error.size());

            isNewObservations = true;

            auto n = observedValues.size();
            auto m = x.size();

            VectorType newObservedValues = VectorType(n + m);
            newObservedValues.head(n) = observedValues;

            VectorType newObservedValueErrors = VectorType(n + m);
            newObservedValueErrors.head(n) = observedValueErrors;

            MatrixType newObservedCovariance = MatrixType::Zero(n + m, n + m);
            
            newObservedCovariance.block(0, 0, n, n) = observedCovariance; // hopefully saves some computation time
            newObservedCovariance.block(0, n, n, m) = kernel(observedInputs, x);
            newObservedCovariance.block(n, 0, m, n) = newObservedCovariance.block(0, n, n, m).transpose();
            newObservedCovariance.block(n, n, m, m) = kernel(x, x);
            //newObservedCovariance.block(n, n, m, m).diagonal().array() += 1.e-5;

            for (size_t i = 0; i < m; ++i) {
                observedInputs.push_back(x[i]);
                newObservedValues(n + i) = y[i];
                newObservedValueErrors(n + i) = error[i];
            }

            newObservedCovariance.block(n, n, m, m) += Eigen::MatrixXd(newObservedValueErrors.tail(m).asDiagonal());

#ifndef NDEBUG
            MatrixType control = kernel(observedInputs, observedInputs);
            control += Eigen::MatrixXd(newObservedValueErrors.asDiagonal());
            //control.diagonal().array() += 1.e-5;
            control -= newObservedCovariance;
            assert((control.size() == 0 || control.cwiseAbs().maxCoeff() < 1.e-5));
#endif

            observedValues = newObservedValues;
            observedValueErrors = newObservedValueErrors;
            observedCovariance = newObservedCovariance;
            //observedCovariance = observedCovariance;

            sqrtObservedCovariance = observedCovariance.llt().matrixL();
#ifndef NDEBUG
            assert(sqrtObservedCovariance.isLowerTriangular() && "error computing the cholesky factorization of the observed covariance, check code.");
#endif
        }

        void addObservations(const std::vector<VectorType>& x, const std::vector<double>& y) {
            addObservations(x, y, std::vector<double>(y.size(), 0));
        }

        void addObservation(const VectorType& x, double y, double error = 0) {
            addObservations({x}, {y}, {error});
        }

        const VectorType& getPosteriorMean() { 
            computePosterior(storedInputs);
            return posteriorMean; 
        }

        const MatrixType& getPosteriorCovariance() { 
            computePosterior(storedInputs);
            return posteriorCovariance; 
        }

        const MatrixType& getSqrtPosteriorCovariance() { 
            computePosterior(storedInputs);
            return sqrtPosteriorCovariance; 
        }

        const std::vector<VectorType>& getObservedInputs() const { return observedInputs; }
        const VectorType& getObservedValues() const { return observedValues; }
        const VectorType& getObservedValueErrors() const { return observedValueErrors; }

    private:
        std::function<double (VectorType)> priorMeanFunction;
        Kernel kernel;

        std::vector<VectorType> storedInputs;
        VectorType inputPriorMean;
        //VectorType inputAggregatedErrors;
        MatrixType observationInputCovariance;
        MatrixType inputCovariance;

        VectorType posteriorMean;
        MatrixType posteriorCovariance;
        MatrixType sqrtPosteriorCovariance;

        bool isNewObservations;
        MatrixType observedCovariance;
        MatrixType sqrtObservedCovariance;
        
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

        //VectorType aggregateErrors(const std::vector<VectorType>& x) {
        //    VectorType errors(x.size());
        //    for (size_t i = 0; i < x.size(); ++i) {
        //        double mean = 0, error = 0;
        //        size_t count = 0;
        //        
        //        for (size_t j = 0; j < observedInputs.size(); ++j) {
        //            if (x[i] == observedInputs[j]) {
        //                ++count;
        //                mean += observedValues(j);
        //                error += std::pow(observedValueErrors(j), 2) + std::pow(observedValues(j), 2);
        //            }
        //        }

        //        if (count > 0) {
        //            mean /= count;
        //            error /= count;
        //            error -= std::pow(mean, 2);
        //        }

        //        assert(error >= 0);
        //        errors(i) = error;
        //    }
        //    return errors;
        //}
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
