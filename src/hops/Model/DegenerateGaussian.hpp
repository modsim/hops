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
                           std::vector<long> inactive = std::vector<long>(0));

        [[nodiscard]] MatrixType::Scalar computeNegativeLogLikelihood(const VectorType &x) override;

        [[nodiscard]] std::optional<VectorType> computeLogLikelihoodGradient(const VectorType &x) override;

        std::optional<MatrixType> computeExpectedFisherInformation(const VectorType &x) override;

        bool hasConstantExpectedFisherInformation() override;

        [[nodiscard]] const VectorType &getMean() const;

        [[nodiscard]] const MatrixType &getCovariance() const;

        const std::vector<long> &getInactive() const;

        [[nodiscard]] std::unique_ptr<Model> copyModel() const override;

        std::vector<std::string> getDimensionNames() const override;

    private:
        std::optional<Gaussian> gaussian;
        std::vector<long> inactive;

        void removeRow(MatrixType &matrix, unsigned int rowToRemove) const;

        void removeColumn(MatrixType &matrix, unsigned int colToRemove) const;

        void removeRow(VectorType &vector, unsigned int rowToRemove) const;

        void stripInactive(MatrixType &matrix) const;

        void stripInactive(VectorType &vector) const;
    };
}

#endif //HOPS_DEGENERATEGAUSSIAN_HPP
