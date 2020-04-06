#ifndef HOPS_COLDNESSATTRIBUTE_HPP
#define HOPS_COLDNESSATTRIBUTE_HPP

#include <string>

namespace hops {
    /**
     * @brief Mixin to add coldness to target distribution.
     * @details The result from calculateNegativeLogLikelihood function is adjusted by the coldness.
     * @tparam MarkovChainImpl
     */
    template<typename MarkovChainImpl>
    class ColdnessAttribute : public MarkovChainImpl {
    public:
        ColdnessAttribute(MarkovChainImpl &markovChainImpl, double coldness) : // NOLINT(cppcoreguidelines-pro-type-member-init)
                MarkovChainImpl(markovChainImpl) {
            setColdness(coldness);
        }

        double calculateNegativeLogLikelihood(const typename MarkovChainImpl::StateType &state) {
            return coldness * MarkovChainImpl::calculateNegativeLogLikelihood(state);
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
