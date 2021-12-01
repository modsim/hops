#ifndef HOPS_PROPOSAL_HPP
#define HOPS_PROPOSAL_HPP

#include <any>

#include <hops/RandomNumberGenerator/RandomNumberGenerator.hpp>
#include <hops/Utility/VectorType.hpp>

#include "ProposalParameterName.hpp"


namespace hops {
    class Proposal {
    public:
        /**
         * @Brief Proposes new state and returns the log probability of accepting the new state (Detailed Balance) and
         * the new state.
         */
        virtual std::pair<double, VectorType> propose(RandomNumberGenerator &rng) = 0;

        /**
         * @Brief Accepts latest proposal as new state and then returns new state.
         * @Detailed Might use optimizations with internal data to speed up accepting proposal. Therefore,
         * it has no input data, because all data is moved internally.
         */
        virtual VectorType acceptProposal() = 0;

        /**
         * @Brief Sets new state to start from. Useful for resuming sampling. DO NOT use it to accept a proposal!
         * @Detailed setState can not do internal optimizations when using it to set the proposal. Therefore, never use
         * it to set an accepted proposal. Use acceptProposal() instead.
         */
        virtual void setState(VectorType state) = 0;

        [[nodiscard]] virtual VectorType getState() const = 0;

        [[nodiscard]] virtual VectorType getProposal() const = 0;

//      TODO include and implement
//        [[nodiscard]] virtual std::vector<std::string> getParameterNames() const = 0;
//
//        [[nodiscard]] virtual std::any getParameter(const std::string& name) const = 0;
//
//        [[nodiscard]] virtual std::string getParameterType(const std::string& name) const = 0;
//
//        /**
//         * @sets parameter with value. Throws exception if any contains incompatible type for parameter
//         */
        virtual void setParameter(ProposalParameterName parameterName, const std::any &value) = 0;


        /**
         * @Brief Returns whether underlying implementation has step size. Useful because tuning should be skipped
         * if it doesn't have a step size.
         */
        [[nodiscard]] virtual bool hasStepSize() const = 0;

        /**
         * @Brief Returns name of proposal class.
         */
        [[nodiscard]] virtual std::string getProposalName() const = 0;

        /**
         * @brief Returns the negative log likelihood value associated with the current state.
         * @details This function is only useful, if the underlying proposal implementation has access to the model.
         * If the proposal implementation does not have access, it returns 0.
         */
        [[nodiscard]] virtual double getNegativeLogLikelihood() const {
            return 0.;
        };

        [[nodiscard]] virtual bool hasNegativeLogLikelihood() const {
            return false;
        };

        [[nodiscard]] virtual std::unique_ptr<Proposal> deepCopy() const = 0;

        virtual ~Proposal() = default;
    };
}

#endif //HOPS_PROPOSAL_HPP
