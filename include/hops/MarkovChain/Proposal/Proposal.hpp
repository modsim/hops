#ifndef HOPS_PROPOSAL_HPP
#define HOPS_PROPOSAL_HPP

#include <hops/RandomNumberGenerator/RandomNumberGenerator.hpp>
#include <hops/Utility/VectorType.hpp>


namespace hops {
    class Proposal {
    public:
        /**
         * @Brief Proposes new state and returns the log probability of accepting the new state (Detailed Balance) and
         * the new state.
         */
        virtual std::pair<double, VectorType> propose(RandomNumberGenerator& rng) = 0;

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

        /**
         * @Brief Returns step size if proposal mechanism has one, std::nullopt otherwise.
         */
        [[nodiscard]] virtual std::optional<double> getStepSize() const = 0;

        /**
         * @Brief Sets step size if proposal mechanism has one, does nothing otherwise.
         */
        virtual void setStepSize(double stepSize) = 0;

        /**
         * @Brief Returns whether underlying implementation has step size. Useful because tuning should be skipped
         * if it doesn't have a step size.
         */
         virtual bool hasStepSize() = 0;

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

        virtual ~Proposal() = default;
    };
}

#endif //HOPS_PROPOSAL_HPP
