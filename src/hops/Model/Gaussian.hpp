#ifndef HOPS_GAUSSIAN_HPP
#define HOPS_GAUSSIAN_HPP

#include <stdexcept>
#define _USE_MATH_DEFINES

#include <math.h> // Using deprecated math for windows
#include <Eigen/Cholesky>
#include <Eigen/Core>
#include <Eigen/LU>
#include <utility>
#include <hops/Model/Model.hpp>

namespace hops {
    class Gaussian : public Model {
    public:
        Gaussian(VectorType mean, MatrixType covariance);

        /**
         * @brief Evaluates the negative log likelihood for input x.
         * @param x
         * @return
         */
        [[nodiscard]] MatrixType::Scalar computeNegativeLogLikelihood(const VectorType &x) override;

        [[nodiscard]] std::optional<VectorType> computeLogLikelihoodGradient(const VectorType &x) override;

        [[nodiscard]] std::optional<MatrixType> computeExpectedFisherInformation(const VectorType &) override;

        bool hasConstantExpectedFisherInformation() override;

        [[nodiscard]] const VectorType &getMean() const;

        [[nodiscard]] const MatrixType &getCovariance() const;

        [[nodiscard]] std::unique_ptr<Model> copyModel() const override;

        std::vector<std::string> getDimensionNames() const override;

    private:
        VectorType mean;
        MatrixType covariance;
        MatrixType inverseCovariance;
        typename MatrixType::Scalar logNormalizationConstant;
    };

    Gaussian::Gaussian(VectorType mean, MatrixType covariance) :
            mean(std::move(mean)),
            covariance(std::move(covariance)) {
        if (mean.size() != covariance.rows()) 
            throw std::runtime_error("Dimension mismatch between mean (dim=" + 
                std::to_string(mean.size()) + ") and covariance (dim=" + 
                std::to_string(covariance.size()) + ").");

        Eigen::LLT<MatrixType> solver(this->covariance);
        if (!this->covariance.isApprox(this->covariance.transpose()) || solver.info() == Eigen::NumericalIssue) {
            throw std::domain_error(
                    "Possibly non semi-positive definite covariance in initialization for Gaussian.");
        }
        Eigen::Matrix<typename MatrixType::Scalar, Eigen::Dynamic, Eigen::Dynamic> matrixL = solver.matrixL();
        Eigen::Matrix<typename MatrixType::Scalar, Eigen::Dynamic, Eigen::Dynamic> inverseMatrixL = matrixL.inverse();
        inverseCovariance = inverseMatrixL.transpose() * inverseMatrixL;

        logNormalizationConstant = -static_cast<typename MatrixType::Scalar>(this->mean.rows()) / 2 *
                                   std::log(2 * M_PI)
                                   - matrixL.diagonal().array().log().sum();
    }

    typename MatrixType::Scalar
    Gaussian::computeNegativeLogLikelihood(const VectorType &x) {
        if (x.size() != mean.size()) 
            throw std::runtime_error("Dimension mismatch between input x (dim=" + 
                std::to_string(x.size()) + ") and Gaussian (dim=" + 
                std::to_string(mean.size()) + ").");
        return -logNormalizationConstant +
               0.5 * static_cast<typename MatrixType::Scalar>((x - mean).transpose() * inverseCovariance * (x - mean));
    }

    std::optional<VectorType> Gaussian::computeLogLikelihoodGradient(const VectorType &x) {
        if (x.size() != mean.size()) 
            throw std::runtime_error("Dimension mismatch between input x (dim=" + 
                std::to_string(x.size()) + ") and Gaussian (dim=" + 
                std::to_string(mean.size()) + ").");
        return -inverseCovariance * (x - mean);
    }

    std::optional<MatrixType> Gaussian::computeExpectedFisherInformation(const VectorType &) {
        return inverseCovariance;
    }

    const VectorType &Gaussian::getMean() const {
        return mean;
    }

    const MatrixType &Gaussian::getCovariance() const {
        return covariance;
    }

    std::unique_ptr<Model> Gaussian::copyModel() const {
        return std::make_unique<Gaussian>(mean, covariance);
    }

    std::vector<std::string> Gaussian::getDimensionNames() const {
        std::vector<std::string> names;
        for (long i = 0; i < mean.rows(); ++i) {
            names.emplace_back("x_" + std::to_string(i));
        }
        return names;
    }

    bool Gaussian::hasConstantExpectedFisherInformation() {
        return true;
    }
}

#endif //HOPS_GAUSSIAN_HPP
