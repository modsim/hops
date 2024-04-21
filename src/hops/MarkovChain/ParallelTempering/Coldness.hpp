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
            return m_coldness == 0 ? 0. : m_coldness * ModelType::computeNegativeLogLikelihood(state);
        }

        [[nodiscard]] std::optional<VectorType> computeLogLikelihoodGradient(const VectorType &state) {
            if (m_coldness != 0) {
                auto gradient = ModelType::computeLogLikelihoodGradient(state);
                if (gradient) {
                    return m_coldness * gradient.value();
                }
            }
            return std::nullopt;
        }

        [[nodiscard]] std::optional<MatrixType> computeExpectedFisherInformation(const VectorType &state) {
            if (m_coldness != 0) {
                auto fisherInformation = ModelType::computeExpectedFisherInformation(state);
                if (fisherInformation) {
                    // Scales quadratically with coldness, because FI is
                    // Jacobian^T * measurementMatrix^-1 * Jacobian
                    return m_coldness * m_coldness * fisherInformation.value();
                }
            }
            return std::nullopt;
        }

        [[nodiscard]] double getColdness() const {
            return m_coldness;
        }

        void setColdness(double newColdness) {
            if (newColdness > 1) {
                m_coldness = 1;
            } else if (newColdness < 0) {
                m_coldness = 0;
            } else {
                m_coldness = newColdness;
            }
        }

    private:
        double m_coldness = 1;
    };
}

#endif //HOPS_COLDNESS_HPP
