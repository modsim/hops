#include "DegenerateGaussian.hpp"

void hops::DegenerateGaussian::removeRow(hops::MatrixType &matrix, unsigned int rowToRemove) const {
    unsigned int numRows = matrix.rows() - 1;
    unsigned int numCols = matrix.cols();

    if (rowToRemove < numRows) {
        matrix.block(rowToRemove, 0, numRows - rowToRemove, numCols) =
                matrix.bottomRows(numRows - rowToRemove);
    }

    matrix.conservativeResize(numRows, numCols);
}

void hops::DegenerateGaussian::removeColumn(hops::MatrixType &matrix, unsigned int colToRemove) const {
    unsigned int numRows = matrix.rows();
    unsigned int numCols = matrix.cols() - 1;

    if (colToRemove < numCols) {
        matrix.block(0, colToRemove, numRows, numCols - colToRemove) =
                matrix.rightCols(numCols - colToRemove);
    }

    matrix.conservativeResize(numRows, numCols);
}

void hops::DegenerateGaussian::removeRow(hops::VectorType &vector, unsigned int rowToRemove) const {
    unsigned int numRows = vector.rows() - 1;

    if (rowToRemove < numRows) {
        vector.segment(rowToRemove, numRows - rowToRemove) = vector.tail(numRows - rowToRemove);
    }

    vector.conservativeResize(numRows);
}

void hops::DegenerateGaussian::stripInactive(hops::MatrixType &matrix) const {
    for (auto &i : inactive) {
        removeRow(matrix, i);
        removeColumn(matrix, i);
    }
}

void hops::DegenerateGaussian::stripInactive(hops::VectorType &vector) const {
    for (auto &i : inactive) {
        removeRow(vector, i);
    }
}

hops::DegenerateGaussian::DegenerateGaussian(VectorType mean,
                                             MatrixType covariance,
                                             std::vector<long> inactive) :
        inactive(std::move(inactive)) {
    if (mean.size() != covariance.rows())
        throw std::runtime_error("Dimension mismatch between mean (dim=" +
                                 std::to_string(mean.size()) + ") and covariance (dim=" +
                                 std::to_string(covariance.size()) + ").");

    stripInactive(mean);
    stripInactive(covariance);
    gaussian = Gaussian(mean, covariance);
}

hops::MatrixType::Scalar
hops::DegenerateGaussian::computeNegativeLogLikelihood(const VectorType &x) {
    VectorType _x = x;
    stripInactive(_x);
    return gaussian.value().computeNegativeLogLikelihood(_x);
}

std::optional<hops::VectorType>
hops::DegenerateGaussian::computeLogLikelihoodGradient(const VectorType &x) {
    VectorType _x = x;
    stripInactive(_x);
    return gaussian.value().computeLogLikelihoodGradient(_x);
}

std::unique_ptr<hops::Model> hops::DegenerateGaussian::copyModel() const {
    return std::make_unique<DegenerateGaussian>(gaussian.value().getMean(),
                                                gaussian.value().getCovariance(),
                                                inactive);
}

const hops::VectorType &hops::DegenerateGaussian::getMean() const {
    return gaussian.value().getMean();
}

const hops::MatrixType &hops::DegenerateGaussian::getCovariance() const {
    return gaussian.value().getCovariance();
}

const std::vector<long> &hops::DegenerateGaussian::getInactive() const {
    return inactive;
}

std::vector<std::string> hops::DegenerateGaussian::getDimensionNames() const {
    std::vector<std::string> names;
    if (gaussian) {
        for (long i = 0; i < gaussian.value().getMean().rows(); ++i) {
            names.emplace_back("x_" + std::to_string(i));
        }
    }
    return names;
}

std::optional<hops::MatrixType> hops::DegenerateGaussian::computeExpectedFisherInformation(const VectorType &x) {
    if (this->inactive.empty()) {
        return gaussian.value().computeExpectedFisherInformation(x);
    }
    return std::nullopt;
}

bool hops::DegenerateGaussian::hasConstantExpectedFisherInformation() {
    return true;
}

