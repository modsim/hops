#ifndef HOPS_COLDNESS_HPP
#define HOPS_COLDNESS_HPP

#include <optional>
#include <utility>

#include "hops/Utility/MatrixType.hpp"
#include "hops/Utility/VectorType.hpp"


namespace hops {
    /**
     * @brief Mixin that adds coldness to model evaluations.
     * @details The result of calls to model are rescaled the coldness. This is useful for ensemble models
     *          such as Parallel Tempering.
     */
    template<typename ModelType>
    class Coldness : public ModelType {
    public:
        explicit Coldness(ModelType model, double coldness = 1) :
                ModelType(std::move(model)) {
            setColdness(coldness);
        }

        [[nodiscard]] MatrixType::Scalar computeNegativeLogLikelihood(const VectorType &state) {
            return coldness == 0 ? 0. : coldness * ModelType::computeNegativeLogLikelihood(state);
        }

        [[nodiscard]] std::optional<VectorType> computeLogLikelihoodGradient(const VectorType &state) {
            if (coldness != 0) {
                auto gradient = ModelType::computeLogLikelihoodGradient(state);
                if (gradient) {
                    return coldness * gradient.value();
                }
            }
            return std::nullopt;
        }

        [[nodiscard]] std::optional<MatrixType> computeExpectedFisherInformation(const VectorType &state) {
            if (coldness != 0) {
                auto fisherInformation = ModelType::computeExpectedFisherInformation(state);
                if (fisherInformation) {
                    // Scales quadratically with coldness, because FI is
                    // Jacobian^T * measurementMatrix^-1 * Jacobian
                    return coldness * coldness * fisherInformation.value();
                }
            }
            return std::nullopt;
        }

        [[nodiscard]] double getColdness() const {
            return coldness;
        }

        void setColdness(double newColdness) {
            if (newColdness > 1) {
                coldness = 1;
            } else if (newColdness < 0) {
                coldness = 0;
            } else {
                coldness = newColdness;
            }
        }

    private:
        double coldness = 1;
    };
}

#endif //HOPS_COLDNESS_HPP
