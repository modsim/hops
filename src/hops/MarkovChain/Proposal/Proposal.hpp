#ifndef HOPS_PROPOSAL_HPP
#define HOPS_PROPOSAL_HPP

#include <algorithm>
#include <any>
#include <memory>
#include <optional>
#include <stdexcept>

#include "hops/RandomNumberGenerator/RandomNumberGenerator.hpp"
#include "hops/Utility/MatrixType.hpp"
#include "hops/Utility/VectorType.hpp"

#include "ProposalParameter.hpp"


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
         * @param activeIndices vector should contain true active parameters and 0 for inactive parameters.
         */
        virtual VectorType &propose(RandomNumberGenerator &rng, const VectorType &activSubspace) {
            throw std::runtime_error("Proposal::Propose with rng and activeIndices not implemented");
        };

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

        virtual void setProposal(const VectorType &newProposal) {
            // expensive default implementation. Replace by specialized implementation where required.
            RandomNumberGenerator rng(0);
            VectorType &internalProposal = this->propose(rng);
            internalProposal = newProposal;
        }

        [[nodiscard]] virtual VectorType getState() const = 0;

        [[nodiscard]] virtual VectorType getProposal() const = 0;

        /**
         * @brief set names for each dimension of the state space. Should typically be set from the Model to be sampled.
         */
        virtual void setDimensionNames(const std::vector<std::string> &names) = 0;

        /**
         * @return names for each dimension of the state space
         */
        [[nodiscard]] virtual std::vector<std::string> getDimensionNames() const = 0;

        /**
         * @brief returns names of parameters of the underlying proposal implementation.
         * @return
         */
        [[nodiscard]] virtual std::vector<std::string> getParameterNames() const = 0;

        [[nodiscard]] virtual std::optional<double> getStepSize() const {
            return std::nullopt;
        };

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
         * @Brief Returns name of proposal class.
         */
        [[nodiscard]] virtual std::string getProposalName() const = 0;

        /**
         * @brief Returns the negative log likelihood value associated with the current state.
         * @details This function is only useful, if the underlying proposal implementation has access to the model.
         * If the proposal implementation does not have access, it returns 0.
         */
        [[nodiscard]] virtual double getStateNegativeLogLikelihood() {
            return 0.;
        };

        /**
         * @brief Returns the negative log likelihood value associated with the currently proposed state.
         * @details This function is only useful, if the underlying proposal implementation has access to the model.
         * If the proposal implementation does not have access, it returns 0.
         */
        [[nodiscard]] virtual double getProposalNegativeLogLikelihood() {
            return 0.;
        };

        [[nodiscard]] virtual bool hasNegativeLogLikelihood() const {
            return false;
        };

        /**
         * @Brief returns whether proposal is symmetric or not.
         */
        [[nodiscard]] virtual bool isSymmetric() const {
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
