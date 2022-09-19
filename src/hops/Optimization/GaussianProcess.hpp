#ifndef HOPS_GAUSSIANPROCESS_HPP
#define HOPS_GAUSSIANPROCESS_HPP

#include <cmath>
#include <Eigen/Cholesky>
#include <Eigen/Core>
#include <Eigen/SVD>
#include <iostream>
#include <random>
#include <vector>


#include "hops/RandomNumberGenerator/RandomNumberGenerator.hpp"


namespace hops {
    namespace internal{
        template<typename MatrixType>
        MatrixType append(const MatrixType& rhs, const MatrixType& lhs) {
            MatrixType newRhs(rhs.rows() + lhs.rows(), lhs.cols());
            if (rhs.size() > 0) {
                newRhs << rhs, lhs;
            } else {
                newRhs << lhs;
            }
            return newRhs;
        }

        template<typename MatrixType, typename VectorType>
        MatrixType append(const MatrixType& rhs, const VectorType& lhs) {
            MatrixType newRhs(rhs.rows() + 1, lhs.cols());
            if (rhs.size() > 0) {
                newRhs << rhs, lhs.transpose();
            } else {
                newRhs << lhs.transpose();
            }
            return newRhs;
        }

        template<typename VectorType>
        VectorType append(const VectorType& rhs, double lhs) {
            VectorType newRhs(rhs.rows() + 1);
            if (rhs.size() > 0) {
                newRhs << rhs, lhs;
            } else {
                newRhs << lhs;
            }
            return newRhs;
        }
    }

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

        /**
         *  Compute the posterior mean and covariance for the given input. This method checks, if any data has changed before 
         *  recomputing depending quantities, which makes it rather safe to call it before sampling or querying any posterior data
         *
         */
        void computePosterior(const MatrixType& input) {
            bool isNewInput = (!storedInputs.size() || input != storedInputs);
            if (isNewObservations || isNewInput) {
                observationInputCovariance = kernel(observedInputs, input);
                //inputAggregatedErrors = aggregateErrors(input);
                
                if (isNewInput) {
                    inputCovariance = kernel(input, input);
                    inputPriorMean = priorMean(input);

                    storedInputs = input;
                }

                if (isNewObservations) {
                    // compute the cholesky factorization on the new observed covariance
                    sqrtObservedCovariance = observedCovariance.llt().matrixL();

                    assert(sqrtObservedCovariance.isLowerTriangular() 
                            && "error computing the cholesky factorization of the observed covariance, check code.");

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
        Eigen::VectorXd sample(const MatrixType& input, 
                               hops::RandomNumberGenerator& randomNumberGenerator, 
                               size_t& maxElement) {
            computePosterior(input); 

            VectorType draw(input.rows());
            auto standardNormal = std::normal_distribution<double>();

            for (long i = 0; i < input.rows(); ++i) {
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

        Eigen::VectorXd sample(const MatrixType& x, 
                               hops::RandomNumberGenerator& randomNumberGenerator) {
            size_t max;
            return sample(x, randomNumberGenerator, max);
        }

        Eigen::VectorXd sample(hops::RandomNumberGenerator& randomNumberGenerator) {
            size_t max;
            return sample(storedInputs, randomNumberGenerator, max);
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
        std::tuple<MatrixType, MatrixType, VectorType> updateObservations(const MatrixType& x, 
                                                                          const VectorType& y, 
                                                                          const VectorType& error, 
                                                                          bool isUnique = false) {
            assert(x.size() == y.size());
            assert(y.size() == error.size());

            // collect indices of observations which should be updated
            std::vector<long> updateCandidates;
            std::vector<long> notFound;
            for (long i = 0; i < x.rows(); ++i) {
                bool foundInput = false;
                for (long j = 0; j < observedInputs.rows(); ++j) {
                    if (x.row(i) == observedInputs.row(j)) {
                        updateCandidates.push_back(j);
                        foundInput = true;

                        if (isUnique) {
                            break;
                        }
                    }
                }

                // record data which could not be updated, because it was not stored in *this the first place
                // this data is "returned" in the reference arguments, such that it can be added after this function returns
                if (!foundInput) {
                    notFound.push_back(i);
                }
            }

            // if any entries were updated, set the isNewObservations flag
            if (updateCandidates.size() > 0) {
                isNewObservations = true;
            }

            for (size_t i = 0; i < updateCandidates.size(); ++i) {
                size_t k = updateCandidates[i];

                // update the data
                observedValues(k) = y(i);
                observedValueErrors(k) = error(i);

                // update the observed covariance 
                //observedCovariance.row(k) = kernel({observedInputs[k]}, observedInputs);
                //observedCovariance.col(k) = observedCovariance.row(k).transpose();
                observedCovariance(k, k) = kernel(observedInputs.row(k), observedInputs.row(k))(0, 0) + observedValueErrors(k);
                //observedCovariance(k, k) += 1.e-5;
            }

//#ifndef NDEBUG
//            MatrixType control = kernel(observedInputs, observedInputs);
//            control -= observedCovariance;
//            assert(control.size() == 0 || control.cwiseAbs().maxCoeff() < 1.e-5);
//#endif

            MatrixType inputsNotFound(notFound.size(), x.cols());
            VectorType valuesNotFound(notFound.size());
            VectorType errorsNotFound(notFound.size());

            for (size_t i = 0; i < notFound.size(); ++i) {
                long k = notFound[i];

                inputsNotFound.row(i) = x.row(k);
                valuesNotFound.row(i) = y.row(k);
                errorsNotFound.row(i) = error.row(k);
            }

            //x = inputsNotFound;
            //y = valuesNotFound;
            //error = errorsNotFound;
            return {inputsNotFound, valuesNotFound, errorsNotFound};
        }

        //void updateObservations(MatrixType& x, 
        //                        VectorType& y, 
        //                        VectorType& error, 
        //                        bool isUnique = false) {
        //    std::tie(x, y, error) = updateObservationsConstArgs(x, y, error);
        //}

        void updateObservations(const MatrixType& x, const VectorType& y) {
            updateObservations(x, y, VectorType::Zeros(y.rows()));
        }

        //void updateObservation(const VectorType& x, double y, double error = 0) {
        //    updateObservations({x}, {y}, {error});
        //}

        void addObservations(const MatrixType& x, VectorType& y, VectorType& error) {
            assert(x.size() == y.size());
            assert(y.size() == error.size());

            isNewObservations = true;

            auto n = observedValues.size();
            auto m = x.size();

            //VectorType newObservedValues = VectorType(n + m);
            //newObservedValues << observedValues, y;
            observedValues = internal::append(observedValues, y);

            //VectorType newObservedValueErrors = VectorType(n + m);
            //newObservedValueErrors << observedValueErrors, error;
            observedValueErrors = internal::append(observedValueErrors, error);

            MatrixType newObservedCovariance = MatrixType::Zero(n + m, n + m);
            
            newObservedCovariance.block(0, 0, n, n) = observedCovariance; // hopefully saves some computation time
            newObservedCovariance.block(0, n, n, m) = kernel(observedInputs, x);
            newObservedCovariance.block(n, 0, m, n) = newObservedCovariance.block(0, n, n, m).transpose().eval();
            newObservedCovariance.block(n, n, m, m) = kernel(x, x);
            //newObservedCovariance.block(n, n, m, m).diagonal().array() += 1.e-5;

            newObservedCovariance.block(n, n, m, m) += MatrixType(error.asDiagonal());

            //MatrixType newObservedInputs = MatrixType::Zero(n + m, n + m);
            //newObservedInputs << observedInputs, x;
            observedInputs = internal::append(observedInputs, x);
            
#ifndef NDEBUG
            MatrixType control = kernel(observedInputs, observedInputs);
            control += Eigen::MatrixXd(observedValueErrors.asDiagonal());
            //control.diagonal().array() += 1.e-5;
            control -= newObservedCovariance;
            assert((control.size() == 0 || control.cwiseAbs().maxCoeff() < 1.e-5));
#endif

            //observedInputs = newObservedInputs;
            //observedValues = newObservedValues;
            //observedValueErrors = newObservedValueErrors;
            observedCovariance = newObservedCovariance;
            //observedCovariance = observedCovariance;
        }

        void addObservations(const MatrixType& x, const VectorType& y) {
            addObservations(x, y, VectorType::Zeros(y.rows()));
        }

        //void addObservation(const VectorType& x, double y, double error = 0) {
        //    addObservations({x}, {y}, {error});
        //}

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

        const MatrixType& getObservedInputs() const { return observedInputs; }
        const VectorType& getObservedValues() const { return observedValues; }
        const VectorType& getObservedValueErrors() const { return observedValueErrors; }

        const MatrixType& getObservedCovariance() const { return observedCovariance; }

        std::function<double (VectorType)>& getPriorMeanFunction() {
            return priorMeanFunction;
        }

        void setKernelSigma(double sigma) {
            double oldSigma = kernel.sigma;
            kernel.sigma = sigma;
            observedCovariance -= MatrixType(observedValueErrors.asDiagonal());
            observedCovariance.array() *= (sigma / oldSigma);
            observedCovariance += MatrixType(observedValueErrors.asDiagonal());
        }

        Kernel getKernel() {
            return kernel;
        }

    private:
        std::function<double (VectorType)> priorMeanFunction;
        Kernel kernel;

        MatrixType storedInputs;
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
        
        MatrixType observedInputs;
        VectorType observedValues;
        VectorType observedValueErrors;

        VectorType priorMean(const MatrixType& x) {
            VectorType prior(x.rows());
            for (long i = 0; i < x.rows(); ++i) {
                prior(i) = priorMeanFunction(x.row(i));
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
}

#endif // HOPS_GAUSSIANPROCESS_HPP
