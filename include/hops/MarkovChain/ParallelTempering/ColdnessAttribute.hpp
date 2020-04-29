#ifndef HOPS_COLDNESSATTRIBUTE_HPP
#define HOPS_COLDNESSATTRIBUTE_HPP

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
        ColdnessAttribute(const Model &markovChainImpl, double coldness = 1) : // NOLINT(cppcoreguidelines-pro-type-member-init)
                Model(markovChainImpl) {
            setColdness(coldness);
        }

        typename Model::VectorType::Scalar calculateNegativeLogLikelihood(const typename Model::VectorType &state) {
            return coldness * Model::calculateNegativeLogLikelihood(state);
        }

        [[nodiscard]] double getColdness() const {
            return coldness;
        }

        void setColdness(double newColdness) {
            if (newColdness > 1) {
                coldness = 1;
            } else if (newColdness < 0) {
                coldness = 0;
            }
            else {
                coldness = newColdness;
            }
        }

    private:
        double coldness;
    };
}

#endif //HOPS_COLDNESSATTRIBUTE_HPP
