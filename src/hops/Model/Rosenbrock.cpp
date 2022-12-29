#include "Rosenbrock.hpp"

hops::Rosenbrock::Rosenbrock(double scaleParameter,
                             VectorType shiftParameter) :
        scaleParameter(scaleParameter),
        shiftParameter(std::move(shiftParameter)) {
    numberOfDimensions = this->shiftParameter.rows() * 2;
}

typename hops::MatrixType::Scalar
hops::Rosenbrock::computeNegativeLogLikelihood(const VectorType &x) {
    if (x.size() != numberOfDimensions)
        throw std::runtime_error("Dimension mismatch between input x (dim=" +
                                 std::to_string(x.size()) + ") and Rosenbrock (dim=" +
                                 std::to_string(numberOfDimensions) + ").");

    typename MatrixType::Scalar result = 0;
    for (long i = 0; i < shiftParameter.rows(); ++i) {
        result += scaleParameter *
                  (100 * std::pow(std::pow(x(2 * i), 2) - x(2 * i + 1), 2) +
                   std::pow(x(2 * i) - shiftParameter(i), 2));
    }
    return result;
}

hops::MatrixType hops::Rosenbrock::computeHessian(const VectorType &x) {
    if (x.size() != numberOfDimensions)
        throw std::runtime_error("Dimension mismatch between input x (dim=" +
                                 std::to_string(x.size()) + ") and Rosenbrock (dim=" +
                                 std::to_string(numberOfDimensions) + ").");
    MatrixType hessian = MatrixType::Zero(x.rows(), x.rows());

    for (long i = 0; i < shiftParameter.rows(); ++i) {
        hessian(2 * i, 2 * i) =
                scaleParameter * (1200 * std::pow(x(2 * i), 2) - 400 * x(2 * i + 1) + 2);
        hessian(2 * i + 1, 2 * i) = scaleParameter * -400 * x(2 * i);
        hessian(2 * i, 2 * i + 1) = scaleParameter * -400 * x(2 * i);
        hessian(2 * i + 1, 2 * i + 1) = scaleParameter * 200;
    }

    return hessian;
}

std::optional<hops::VectorType> hops::Rosenbrock::computeLogLikelihoodGradient(const VectorType &x) {
    if (x.size() != numberOfDimensions)
        throw std::runtime_error("Dimension mismatch between input x (dim=" +
                                 std::to_string(x.size()) + ") and Rosenbrock (dim=" +
                                 std::to_string(numberOfDimensions) + ").");

    VectorType gradient = VectorType::Zero(x.rows());
    for (long i = 0; i < shiftParameter.rows(); ++i) {
        gradient(2 * i) = scaleParameter * (4 * 100 * (x(2 * i + 1) - std::pow(x(2 * i), 2)) * (-2 * x(2 * i)) +
                                            2 * (x(2 * i) - shiftParameter(i)));

        gradient(2 * i + 1) = scaleParameter * 2 * 100 * (x(2 * i + 1) - std::pow(x(2 * i), 2));
    }
    return gradient;
}

std::optional<hops::MatrixType> hops::Rosenbrock::computeExpectedFisherInformation(const VectorType &x) {
    if (x.size() != numberOfDimensions)
        throw std::runtime_error("Dimension mismatch between input x (dim=" +
                                 std::to_string(x.size()) + ") and Rosenbrock (dim=" +
                                 std::to_string(numberOfDimensions) + ").");

    MatrixType hessian = computeHessian(x);
    // regularization should be between 0 and infinity. This value is guessed for now.
    double regularization = 1. / hessian.maxCoeff();
    MatrixType expPositive = (regularization * hessian).exp();
    MatrixType expNegative = (-regularization * hessian).exp();
    hessian = (expPositive + expNegative) * hessian * (expPositive - expNegative).inverse();
    return
            hessian;
}

std::unique_ptr<hops::Model> hops::Rosenbrock::copyModel() const {
    return std::make_unique<Rosenbrock>(scaleParameter, shiftParameter);
}

double hops::Rosenbrock::getScaleParameter() const {
    return scaleParameter;
}

const hops::VectorType &hops::Rosenbrock::getShiftParameter() const {
    return shiftParameter;
}

long hops::Rosenbrock::getNumberOfDimensions() const {
    return numberOfDimensions;
}

std::vector<std::string> hops::Rosenbrock::getDimensionNames() const {
    std::vector<std::string> names;
    for (long i = 0; i < numberOfDimensions; ++i) {
        names.emplace_back("x_" + std::to_string(i));
    }
    return names;
}

