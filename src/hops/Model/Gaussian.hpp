#ifndef HOPS_GAUSSIAN_HPP
#define HOPS_GAUSSIAN_HPP

#define _USE_MATH_DEFINES

#include <math.h> // Using deprecated math for windows
#include <Eigen/Cholesky>
#include <Eigen/Core>
#include <Eigen/LU>
#include <utility>

#include "hops/Model/Model.hpp"

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

        [[nodiscard]] const MatrixType &getCovarianceLowerCholesky() const;

        [[nodiscard]] std::unique_ptr<Model> copyModel() const override;

        [[nodiscard]] std::vector<std::string> getDimensionNames() const override;

    private:
        VectorType mean;
        MatrixType covariance;
        MatrixType covarianceLowerCholesky;
        MatrixType inverseCovariance;
        typename MatrixType::Scalar logNormalizationConstant;
    };
}

#endif //HOPS_GAUSSIAN_HPP
