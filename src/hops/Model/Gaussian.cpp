#include <stdexcept>

#include "Gaussian.hpp"

hops::Gaussian::Gaussian(hops::VectorType mean, hops::MatrixType covariance) :
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
    covarianceLowerCholesky = solver.matrixL();
    Eigen::Matrix<typename MatrixType::Scalar, Eigen::Dynamic, Eigen::Dynamic> inverseMatrixL =
            covarianceLowerCholesky.inverse();
    inverseCovariance = inverseMatrixL.transpose() * inverseMatrixL;

    logNormalizationConstant = -static_cast<typename MatrixType::Scalar>(this->mean.rows()) / 2 *
                               std::log(2 * M_PI)
                               - covarianceLowerCholesky.diagonal().array().log().sum();
}

typename hops::MatrixType::Scalar
hops::Gaussian::computeNegativeLogLikelihood(const hops::VectorType &x) {
    if (x.size() != mean.size())
        throw std::runtime_error("Dimension mismatch between input x (dim=" +
                                 std::to_string(x.size()) + ") and Gaussian (dim=" +
                                 std::to_string(mean.size()) + ").");
    return -logNormalizationConstant +
           0.5 * static_cast<typename MatrixType::Scalar>((x - mean).transpose() * inverseCovariance * (x - mean));
}

std::optional<hops::VectorType> hops::Gaussian::computeLogLikelihoodGradient(const hops::VectorType &x) {
    if (x.size() != mean.size())
        throw std::runtime_error("Dimension mismatch between input x (dim=" +
                                 std::to_string(x.size()) + ") and Gaussian (dim=" +
                                 std::to_string(mean.size()) + ").");
    return -inverseCovariance * (x - mean);
}

std::optional<hops::MatrixType> hops::Gaussian::computeExpectedFisherInformation(const hops::VectorType &) {
    return inverseCovariance;
}

const hops::VectorType &hops::Gaussian::getMean() const {
    return mean;
}

const hops::MatrixType &hops::Gaussian::getCovariance() const {
    return covariance;
}

std::unique_ptr<hops::Model> hops::Gaussian::copyModel() const {
    return std::make_unique<Gaussian>(mean, covariance);
}

std::vector<std::string> hops::Gaussian::getDimensionNames() const {
    std::vector<std::string> names;
    for (long i = 0; i < mean.rows(); ++i) {
        names.emplace_back("x_" + std::to_string(i));
    }
    return names;
}

bool hops::Gaussian::hasConstantExpectedFisherInformation() {
    return true;
}

const hops::MatrixType &hops::Gaussian::getCovarianceLowerCholesky() const {
    return covarianceLowerCholesky;
}

