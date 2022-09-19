#ifndef HOPS_MARKOVCHAINADAPTER_HPP
#define HOPS_MARKOVCHAINADAPTER_HPP

#include "hops/Utility/HopsWithinHopsy.hpp"

#include "MarkovChain.hpp"

namespace hops {
    template<typename MarkovChainImpl>
    class MarkovChainAdapter : public MarkovChain, public MarkovChainImpl {
    public:
        explicit MarkovChainAdapter(MarkovChainImpl markovChainImpl) : MarkovChainImpl(markovChainImpl) {}

        std::pair<double, VectorType> draw(RandomNumberGenerator &randomNumberGenerator, long thinning = 1) override {
            double acceptanceRate = 0;
            for (long i = 0; i < thinning; ++i) {
                ABORTABLE;
                acceptanceRate += MarkovChainImpl::draw(randomNumberGenerator);
            }
            return {acceptanceRate / thinning, MarkovChainImpl::getState()};
        }

        [[nodiscard]] VectorType getState() const override {
            return MarkovChainImpl::getState();
        }

        void setState(const VectorType &state) override {
            MarkovChainImpl::setState(state);
        }

        double getStateNegativeLogLikelihood() override {
            return MarkovChainImpl::getStateNegativeLogLikelihood();
        }

        /**
         * @brief gets proposal parameter. Throws exception if proposal has no parameter parameterName.
         * @details Implementations should list possible parameterNames in the exception message.
         */
        [[nodiscard]] std::any getParameter(const ProposalParameter &parameter) const override {
            return MarkovChainImpl::getParameter(parameter);
        }

        /**
         * @brief sets parameter with value. Throws exception if any contains incompatible type for parameter.
         * @details Implementations should list possible parameterNames in the exception message.
         */
        void setParameter(const ProposalParameter &parameter, const std::any &value) override {
            MarkovChainImpl::setParameter(parameter, value);
        }
    };
}

#endif //HOPS_MARKOVCHAINADAPTER_HPP
