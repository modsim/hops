#ifndef HOPS_MODEL_HPP
#define HOPS_MODEL_HPP

#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "hops/Utility/MatrixType.hpp"
#include "hops/Utility/VectorType.hpp"

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

        [[nodiscard]] virtual std::optional<VectorType> computeLogLikelihoodGradient(const VectorType &) {
            return std::nullopt;
        };

        [[nodiscard]] virtual std::optional<MatrixType> computeExpectedFisherInformation(const VectorType &) {
            return std::nullopt;
        }

        /**
         * @Brief Whether the fisher information is constant as a function of the parameters. This is not generally
         * the case but it is the case for important distributions, such as the gaussian distribution.
         * @details If the expected fisher information is constant, some proposal classes can be sped up considerably.
         */
        [[nodiscard]] virtual bool hasConstantExpectedFisherInformation() {
            return false;
        }

        [[nodiscard]] virtual std::vector<std::string> getDimensionNames() const = 0;

        [[nodiscard]] virtual std::unique_ptr<Model> copyModel() const = 0;
    };
}

#endif //HOPS_MODEL_HPP
