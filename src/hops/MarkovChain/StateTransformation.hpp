#ifndef HOPS_STATETRANSFORMATION_HPP
#define HOPS_STATETRANSFORMATION_HPP

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

        typename MarkovChainImpl::StateType getState() {
            return transformation.apply(MarkovChainImpl::getState());
        }

        typename MarkovChainImpl::StateType getProposal() {
            return transformation.apply(MarkovChainImpl::getProposal());
        }

        void setState(const typename MarkovChainImpl::StateType &state) {
            MarkovChainImpl::setState(transformation.applyInverse(state));
        }

    private:
        Transformation transformation;
    };
}

#endif //HOPS_STATETRANSFORMATION_HPP