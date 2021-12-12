#ifndef HOPS_STATETRANSFORMATION_HPP
#define HOPS_STATETRANSFORMATION_HPP

#include <hops/Utility/VectorType.hpp>

namespace hops {
    /**
     * @brief Mixin for undoing transformations to the Markov chain state.
     * @details Prominent use-case is for dealing with rounding.
     * @tparam MarkovChainImpl
     */
    template<typename MarkovChainImpl, typename TransformationImpl>
    class StateTransformation : public MarkovChainImpl {
    public:
        explicit StateTransformation(const MarkovChainImpl &markovChainImpl, TransformationImpl transformation) :
                MarkovChainImpl(markovChainImpl),
                transformation(transformation) {}

        VectorType getState() const {
            return transformation.apply(MarkovChainImpl::getState());
        }

         VectorType getProposal() const {
            return transformation.apply(MarkovChainImpl::getProposal());
        }

        void setState(const VectorType &state) {
            MarkovChainImpl::setState(transformation.revert(state));
        }

    private:
        TransformationImpl transformation;
    };
}

#endif //HOPS_STATETRANSFORMATION_HPP
