#ifndef HOPS_DEGENERATEGAUSSIAN_HPP
#define HOPS_DEGENERATEGAUSSIAN_HPP

#include <Eigen/Cholesky>
#include <Eigen/Core>
#include <Eigen/LU>

#define _USE_MATH_DEFINES

#include <math.h> // Using deprecated math for windows
#include <utility>

#include "Gaussian.hpp"
#include <hops/Utility/MatrixType.hpp>
#include <hops/Utility/VectorType.hpp>


namespace hops {
    class DegenerateGaussian : public Model {
    public:

        DegenerateGaussian(VectorType mean, MatrixType covariance,
                           std::vector<long> inactive = std::vector<long>(0));

        [[nodiscard]] MatrixType::Scalar computeNegativeLogLikelihood(const VectorType &x) override;

        [[nodiscard]] std::optional<VectorType> computeLogLikelihoodGradient(const VectorType &x) override;

        [[nodiscard]] std::optional<MatrixType> computeExpectedFisherInformation(const VectorType &) override;

        [[nodiscard]] const VectorType &getMean() const;

        [[nodiscard]] const MatrixType &getCovariance() const;

        const std::vector<long> &getInactive() const;

        [[nodiscard]] std::unique_ptr<Model> copyModel() const override;

        std::vector<std::string> getDimensionNames() const override;

    private:
        std::optional<Gaussian> gaussian;
        std::vector<long> inactive;

        void removeRow(Eigen::MatrixXd &matrix, unsigned int rowToRemove) const {
            unsigned int numRows = matrix.rows() - 1;
            unsigned int numCols = matrix.cols();

            if (rowToRemove < numRows) {
                matrix.block(rowToRemove, 0, numRows - rowToRemove, numCols) = 
                    matrix.bottomRows(numRows - rowToRemove);
            }

            matrix.conservativeResize(numRows, numCols);
        }

        void removeColumn(Eigen::MatrixXd &matrix, unsigned int colToRemove) const {
            unsigned int numRows = matrix.rows();
            unsigned int numCols = matrix.cols() - 1;

            if (colToRemove < numCols) {
                matrix.block(0, colToRemove, numRows, numCols - colToRemove) = 
                    matrix.rightCols(numCols - colToRemove);
            }

            matrix.conservativeResize(numRows, numCols);
        }

        void removeRow(Eigen::VectorXd &vector, unsigned int rowToRemove) const {
            unsigned int numRows = vector.rows() - 1;

            if (rowToRemove < numRows) {
                vector.segment(rowToRemove, numRows - rowToRemove) = vector.tail(numRows - rowToRemove);
            }

            vector.conservativeResize(numRows);
        }

        void stripInactive(Eigen::MatrixXd &matrix) const {
            for (auto &i : inactive) {
                removeRow(matrix, i);
                removeColumn(matrix, i);
            }
        }

        void stripInactive(Eigen::VectorXd &vector) const {
            for (auto &i : inactive) {
                removeRow(vector, i);
            }
        }
    };

    DegenerateGaussian::DegenerateGaussian(VectorType mean,
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

    MatrixType::Scalar
    DegenerateGaussian::computeNegativeLogLikelihood(const VectorType &x) {
        VectorType _x = x;
        stripInactive(_x);
        return gaussian.value().computeNegativeLogLikelihood(_x);
    }

    std::optional<MatrixType>
    DegenerateGaussian::computeExpectedFisherInformation(const VectorType &x) {
        // Saves performance and skips stripping x here, because the FIM is constant anyways.
        return gaussian.value().computeExpectedFisherInformation(x);
    }

    std::optional<VectorType>
    DegenerateGaussian::computeLogLikelihoodGradient(const VectorType &x) {
        VectorType _x = x;
        stripInactive(_x);
        return gaussian.value().computeLogLikelihoodGradient(_x);
    }

    std::unique_ptr<Model> DegenerateGaussian::copyModel() const {
        return std::make_unique<DegenerateGaussian>(gaussian.value().getMean(), 
                                                    gaussian.value().getCovariance(),
                                                    inactive);
    }

    const VectorType &DegenerateGaussian::getMean() const {
        return gaussian.value().getMean();
    }

    const MatrixType &DegenerateGaussian::getCovariance() const {
        return gaussian.value().getCovariance();
    }

    const std::vector<long> &DegenerateGaussian::getInactive() const {
        return inactive;
    }

    std::vector<std::string> DegenerateGaussian::getDimensionNames() const {
        std::vector<std::string> names;
        if (gaussian) {
            for (long i = 0; i < gaussian.value().getMean().rows(); ++i) {
                names.emplace_back("x_" + std::to_string(i));
            }
        }
        return names;
    }
}

#endif //HOPS_DEGENERATEGAUSSIAN_HPP
