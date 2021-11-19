#ifndef HOPS_STATETRANSFORMATION_HPP
#define HOPS_STATETRANSFORMATION_HPP

#include <hops/Utility/VectorType.hpp>

namespace hops {
    /**
     * @brief Mixin for undoing transformations to the Markov chain state.
     * @details Prominent use-case is for dealing with rounding.
     * @tparam MarkovChainImpl
     */
    template<typename MarkovChainImpl, typename Transformation>
    class StateTransformation : public MarkovChainImpl {
    public:
        explicit StateTransformation(const MarkovChainImpl &markovChainImpl, Transformation transformation) :
                MarkovChainImpl(markovChainImpl),
                transformation(transformation) {}

        VectorType getState() {
            return transformation.apply(MarkovChainImpl::getState());
        }

         VectorType getProposal() {
            return transformation.apply(MarkovChainImpl::getProposal());
        }

        void setState(const VectorType &state) {
            MarkovChainImpl::setState(transformation.revert(state));
        }

    private:
        Transformation transformation;
    };
}

#endif //HOPS_STATETRANSFORMATION_HPP
