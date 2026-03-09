#ifndef HOPS_DEGENERATEGAUSSIAN_HPP
#define HOPS_DEGENERATEGAUSSIAN_HPP

#include <Eigen/Cholesky>
#include <Eigen/Core>
#include <Eigen/LU>

#define _USE_MATH_DEFINES

#include <math.h> // Using deprecated math for windows
#include <utility>

#include "hops/Utility/MatrixType.hpp"
#include "hops/Utility/VectorType.hpp"

#include "Gaussian.hpp"


namespace hops {
    class DegenerateGaussian : public Model {
    public:

        DegenerateGaussian(VectorType mean, MatrixType covariance,
                           std::vector<Eigen::Index> inactive = std::vector<Eigen::Index>(0));

        [[nodiscard]] MatrixType::Scalar computeNegativeLogLikelihood(const VectorType &x) override;

        [[nodiscard]] std::optional<VectorType> computeLogLikelihoodGradient(const VectorType &x) override;

        std::optional<MatrixType> computeExpectedFisherInformation(const VectorType &x) override;

        bool hasConstantExpectedFisherInformation() override;

        [[nodiscard]] const VectorType &getMean() const;

        [[nodiscard]] const MatrixType &getCovariance() const;

        const std::vector<Eigen::Index> &getInactive() const;

        [[nodiscard]] std::unique_ptr<Model> copyModel() const override;

        std::vector<std::string> getDimensionNames() const override;

    private:
        std::optional<Gaussian> gaussian;
        std::vector<Eigen::Index> inactive;

        void removeRow(MatrixType &matrix, Eigen::Index rowToRemove) const;

        void removeColumn(MatrixType &matrix, Eigen::Index colToRemove) const;

        void removeRow(VectorType &vector, Eigen::Index rowToRemove) const;

        void stripInactive(MatrixType &matrix) const;

        void stripInactive(VectorType &vector) const;
    };
}

#endif //HOPS_DEGENERATEGAUSSIAN_HPP
