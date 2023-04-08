#ifndef HOPS_MARKOVCHAIN_HPP
#define HOPS_MARKOVCHAIN_HPP

#include <memory>
#include <utility>

#include "hops/MarkovChain/Proposal/Proposal.hpp"
#include "hops/MarkovChain/Proposal/ProposalParameter.hpp"
#include "hops/RandomNumberGenerator/RandomNumberGenerator.hpp"
#include "hops/Utility/VectorType.hpp"

namespace hops {
    class MarkovChain {
    public:
        virtual ~MarkovChain() = default;

        /**
         * @brief Updates internal state of the chain and returns a single new state as well as the acceptance rate throughout the thinning.
         * @param randomNumberGenerator
         * @param thinning Number of samples to draw but discard before reporting a single new sample.
         */
        virtual std::pair<double, VectorType> draw(RandomNumberGenerator &randomNumberGenerator, long thinning = 1) = 0;

        [[nodiscard]] virtual VectorType getState() const = 0;

        virtual void setState(const VectorType &) = 0;

        virtual double getStateNegativeLogLikelihood() = 0;

        /**
         * @brief gets proposal parameter. Throws exception if proposal has no parameter parameterName.
         * @details Implementations should list possible parameterNames in the exception message.
         */
        [[nodiscard]] virtual std::any getParameter(const ProposalParameter &parameter) const = 0;

        /**
         * @brief sets parameter with value. Throws exception if any contains incompatible type for parameter.
         * @details Implementations should list possible parameterNames in the exception message.
         */
        virtual void setParameter(const ProposalParameter &parameter, const std::any &value) = 0;
    };
}

#endif //HOPS_MARKOVCHAIN_HPP

