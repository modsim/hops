#ifndef HOPS_PROPOSAL_HPP
#define HOPS_PROPOSAL_HPP

#include <algorithm>
#include <any>

#include <hops/RandomNumberGenerator/RandomNumberGenerator.hpp>
#include <hops/Utility/MatrixType.hpp>
#include <hops/Utility/VectorType.hpp>
#include <stdexcept>

#include "ProposalParameter.hpp"
#include "ProposalStatistics.hpp"


namespace hops {
    class Proposal {
    public:
        /**
         * @Brief Proposes new state and returns it. The returned state can be stored, transformed or use in conjunction with
         * a metropolis-hastings filter and other techniques
         * the new state.
         */
        virtual VectorType &propose(RandomNumberGenerator &rng) = 0;

        /**
         * @Brief Proposes new state on a subspace and returns it. The returned state can be stored, transformed or use in conjunction with
         * a metropolis-hastings filter, reversible jump mcmc and other techniques
         * the new state.
         * @param activeParameters vector should contain true active parameters and 0 for inactive parameters.
         */
        virtual VectorType &propose(RandomNumberGenerator &rng, const  Eigen::VectorXi &activeParameters) {
            throw std::runtime_error("Not implemented");
        };

        /**
         * @brief ProposalStatistics are only useful, when tracking of proposal infos is activated.
         * The infos about a series of proposals should be called before getSate,
         * because the infos are reset, when a proposal is accepted.
         * @return
         */
        [[nodiscard]] virtual ProposalStatistics &getProposalStatistics() {
            throw std::runtime_error("getProposalStatistics is not implemented for this Proposal");
        };

        /**
         * @brief Returns proposalStatistics and resets proposalStatistics to empty statistic
         */
        virtual ProposalStatistics getAndResetProposalStatistics() {
            throw std::runtime_error("getAndResetProposalStatistics is not implemented for this Proposal");
        }

        /**
         * @brief In some settings, like uniform sampling, tracking proposal info is not very useful but also
         * considerably slows down the sampling, therefore it has to be activated when required.
         */
        virtual void activateTrackingOfProposalStatistics() {
            throw std::runtime_error("Not implemented.");
        };

        /**
         * @brief In some settings, like uniform sampling, tracking proposal info is not very useful but also
         * considerably slows down the sampling. This function can be used to disable tracking.
         */
        virtual void disableTrackingOfProposalStatistics() {};

        virtual bool isTrackingOfProposalStatisticsActivated() { return false; }

        /**
         * @Brief Calculates detailed balance using internal proposal. Saves one copy operation compared to
         computeLogAcceptanceProbability(const VectorType& proposal).
         * @detailed Potentially changes internal state, because quantities related to the proposal might have to be evaluated in order to calculate the correct probability.
         */
        [[nodiscard]] virtual double computeLogAcceptanceProbability() = 0;

        /**
         * @Brief Accepts latest proposal as new state and then returns new state.
         * @Detailed Might use optimizations with internal data to speed up accepting proposal. Therefore,
         * it has no input data, because all data is moved internally. The internal data manipulations happen
         * inside of computeLogAcceptanceProbability.
         * WARNING: Potentially destroys internal representation of proposal statistics
         */
        virtual VectorType &acceptProposal() = 0;

        /**
         * @Brief Sets new state to start from. Useful for resuming sampling. DO NOT use it to accept a proposal, as it is computationally heavier than acceptProposal!
         * @Detailed setState can not do internal optimizations when using it to set the proposal. Therefore, never use
         * it to set an accepted proposal. Use acceptProposal() instead.
         */
        virtual void setState(const VectorType &state) = 0;

        [[nodiscard]] virtual VectorType getState() const = 0;

        [[nodiscard]] virtual VectorType getProposal() const = 0;

        /**
         * @return names for each dimension of the state space
         */
        [[nodiscard]] virtual std::vector<std::string> getDimensionNames() const {
            // Default implementation sets names as x_i for dimension i
            std::vector<std::string> names;
            for (long i = 0; i < getState().rows(); ++i) {
                names.emplace_back("x_" + std::to_string(i));
            }
            return names;
        }

        [[nodiscard]] virtual std::vector<std::string> getParameterNames() const = 0;

        [[nodiscard]] virtual std::any getParameter(const ProposalParameter &parameter) const = 0;

        /**
         * @brief returns string representation of parameter type, e.g. double, int, Eigen::MatrixXd.
         * @param name
         * @return
         */
        [[nodiscard]] virtual std::string getParameterType(const ProposalParameter &parameter) const = 0;

        /**
         * @brief sets parameter with value. Throws exception if any contains incompatible type for parameter.
         * @details Implementations should list possible parameterNames in the exception message.
         */
        virtual void setParameter(const ProposalParameter &parameter, const std::any &value) = 0;

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
        [[nodiscard]] virtual double getStateNegativeLogLikelihood() const {
            return 0.;
        };

        /**
         * @brief Returns the negative log likelihood value associated with the currently proposed state.
         * @details This function is only useful, if the underlying proposal implementation has access to the model.
         * If the proposal implementation does not have access, it returns 0.
         */
        [[nodiscard]] virtual double getProposalNegativeLogLikelihood() const {
            return 0.;
        };

        [[nodiscard]] virtual bool hasNegativeLogLikelihood() const {
            return false;
        };

        /**
         * @brief Returns const reference to dense left-hand side operator A from the polytope defining inequality Ax <= b.
         */
        [[nodiscard]] virtual const MatrixType &getA() const = 0;

        /**
         * @brief Returns const reference to dense right-hand side vector b from the polytope defining inequality Ax <= b.
         */
        [[nodiscard]] virtual const VectorType &getB() const = 0;

        [[nodiscard]] virtual std::unique_ptr<Proposal> copyProposal() const = 0;

        virtual ~Proposal() = default;
    };
}

#endif //HOPS_PROPOSAL_HPP
