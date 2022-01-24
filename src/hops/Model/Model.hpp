#ifndef HOPS_MODEL_HPP
#define HOPS_MODEL_HPP

#include <memory>
#include <optional>
#include <string>
#include <vector>

#include <hops/Utility/MatrixType.hpp>
#include <hops/Utility/VectorType.hpp>

namespace hops {
    class Model {
    public:
        virtual ~Model() = default;

        /**
         * @brief Evaluates the negative log likelihood for input x.
         * @param x
         * @return
         */
        [[nodiscard]] virtual typename MatrixType::Scalar computeNegativeLogLikelihood(const VectorType &x) = 0;

        [[nodiscard]] virtual std::optional<VectorType> computeLogLikelihoodGradient(const VectorType &x) {
            return std::nullopt;
        };

        [[nodiscard]] virtual std::optional<MatrixType> computeExpectedFisherInformation(const VectorType &) {
            return std::nullopt;
        }

        [[nodiscard]] virtual std::vector<std::string> getDimensionNames() const = 0;

        [[nodiscard]] virtual std::unique_ptr<Model> copyModel() const = 0;
    };
}

#endif //HOPS_MODEL_HPP
