#ifndef NUPS_STATETRANSFORMATION_HPP
#define NUPS_STATETRANSFORMATION_HPP

namespace nups {
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

    private:
        Transformation transformation;
    };
}

#endif //NUPS_STATETRANSFORMATION_HPP
