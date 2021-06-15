#ifndef HOPS_COLDNESSATTRIBUTE_HPP
#define HOPS_COLDNESSATTRIBUTE_HPP

#include <hops/Model/IsCalculateLogLikelihoodGradientAvailable.hpp>
#include <string>

namespace hops {
    /**
     * @brief Mixin to add coldness to target distribution.
     * @details The result from calculateNegativeLogLikelihood function is adjusted by the coldness.
     * @tparam Model
     */
    template<typename Model>
    class ColdnessAttribute : public Model {
    public:
        explicit ColdnessAttribute(const Model &markovChainImpl, double coldness = 1)
                : // NOLINT(cppcoreguidelines-pro-type-member-init)
                Model(markovChainImpl) {
            setColdness(coldness);
        }

        typename Model::VectorType::Scalar calculateNegativeLogLikelihood(const typename Model::VectorType &state) {
            return coldness == 0 ? 0. : coldness * Model::calculateNegativeLogLikelihood(state);
        }

        typename Model::VectorType calculateLogLikelihoodGradient(const typename Model::VectorType &state) {
            if (IsCalculateLogLikelihoodGradientAvailable<Model>::value) {
                if (coldness == 0) {
                    return Model::VectorType::Zero(state.rows());
                } else {
                    return coldness * Model::calculateLogLikelihoodGradient(state);
                }
            }
            throw std::runtime_error("Gradient called but it is undefined.");
        }

        typename Model::MatrixType calculateExpectedFisherInformation(const typename Model::VectorType &state) {
            if (IsCalculateLogLikelihoodGradientAvailable<Model>::value) {
                if (coldness == 0) {
                    return Model::MatrixType::Zero(state.rows(), state.rows());
                } else {
                    return coldness * Model::calculateExpectedFisherInformation(state);
                }
            }
            throw std::runtime_error("Fisher Information called but it is undefined.");
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
        double coldness;
    };
}

#endif //HOPS_COLDNESSATTRIBUTE_HPP
