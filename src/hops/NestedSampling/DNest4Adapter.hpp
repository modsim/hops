#ifndef HOPS_DNEST4ADAPTER_HPP
#define HOPS_DNEST4ADAPTER_HPP

#include <hops/extern/DNest4.hpp>

#include <hops/MarkovChain/MarkovChain.hpp>

#include "DNest4EnvironmentSingleton.hpp"

namespace hops {

    /**
     * @Brief An adapter that allows the usage of hops sampling algorithms and hops compatible models with DNest4
     */
    class DNest4Adapter {
    public:
        DNest4Adapter();

        DNest4Adapter(const DNest4Adapter &other);

        DNest4Adapter(DNest4Adapter &&other) noexcept;

        DNest4Adapter &operator=(DNest4Adapter other);

        DNest4Adapter &operator=(DNest4Adapter &&other) noexcept;


        /**
         * @Brief generates a state from the prior for DNest4
         * @param rng
         */
        void from_prior(DNest4::RNG &);

        /**
         * @ Metropolis-Hastings proposal for DNest4
         * @param rng
         * @return
         */
        double perturb(DNest4::RNG &);

        void accept_perturbation();

        [[nodiscard]] double log_likelihood() const;

        [[nodiscard]] double proposal_log_likelihood() const;

        /**
         * @Brief Prints current state to stream
         * @param out
         */
        void print(std::ostream &out) const;

        /**
         * @Brief returns string with column names in csv format
         * @return
         */
        [[nodiscard]] std::string description() const;

    private:
        void swap(DNest4Adapter &other) noexcept;

        std::uniform_real_distribution<double> uniformRealDistribution;

        VectorType state;
        VectorType proposal;
        double stateLogAcceptanceProbability = 0;
        double proposalLogAcceptanceProbability = 0;

        std::unique_ptr<hops::Proposal> priorProposer;
        std::unique_ptr<hops::Proposal> posteriorProposer;
        std::unique_ptr<hops::Model> model;

        RandomNumberGenerator internal_rng;
    };

    DNest4Adapter::DNest4Adapter() {
        priorProposer = DNest4EnvironmentSingleton::getInstance().getPriorProposer();
        posteriorProposer = DNest4EnvironmentSingleton::getInstance().getPosteriorProposer();
        model = DNest4EnvironmentSingleton::getInstance().getModel();

        std::random_device seedDevice;
        std::uniform_int_distribution<long> dist(std::numeric_limits<long>::min(),
                                                 std::numeric_limits<long>::max());
        long seed = dist(seedDevice);
        internal_rng.seed(seed);

        state = DNest4EnvironmentSingleton::getInstance().getStartingPoint();
        proposal = state;
        stateLogAcceptanceProbability = 0;
        proposalLogAcceptanceProbability = 0;

    }

    DNest4Adapter::DNest4Adapter(const DNest4Adapter &other) {
        uniformRealDistribution = other.uniformRealDistribution;
        state = other.state;
        proposal = other.proposal;
        stateLogAcceptanceProbability = other.stateLogAcceptanceProbability;
        proposalLogAcceptanceProbability = other.proposalLogAcceptanceProbability;
        priorProposer = other.priorProposer->deepCopy();
        posteriorProposer = other.posteriorProposer->deepCopy();
        model = other.model->deepCopy();
        internal_rng = other.internal_rng;
    }

    DNest4Adapter::DNest4Adapter(DNest4Adapter &&other) noexcept {
        uniformRealDistribution = other.uniformRealDistribution;
        state = std::move(other.state);
        proposal = std::move(other.proposal);
        stateLogAcceptanceProbability = other.stateLogAcceptanceProbability;
        proposalLogAcceptanceProbability = other.proposalLogAcceptanceProbability;
        priorProposer = std::move(other.priorProposer);
        posteriorProposer = std::move(other.posteriorProposer);
        model = std::move(other.model);
        internal_rng = other.internal_rng;
    }

    void DNest4Adapter::from_prior(DNest4::RNG &) {
        auto[logAcceptanceProbability, proposal] = priorProposer->propose(internal_rng);
        double logAcceptanceChance = std::log(uniformRealDistribution(internal_rng));
        if (logAcceptanceChance < logAcceptanceProbability && std::isfinite(logAcceptanceProbability)) {
            this->state = priorProposer->acceptProposal();
        }
    }

    double DNest4Adapter::perturb(DNest4::RNG &) {
        posteriorProposer->setState(state);
        auto posteriorProposal = posteriorProposer->propose(internal_rng);
        proposalLogAcceptanceProbability = posteriorProposal.first;
        proposal = posteriorProposal.second;
        return proposalLogAcceptanceProbability;
    }

    double DNest4Adapter::log_likelihood() const {
        if (std::isfinite(stateLogAcceptanceProbability)) {
            return -model->computeNegativeLogLikelihood(this->state);
        } else {
            return stateLogAcceptanceProbability;
        }
    }

    double DNest4Adapter::proposal_log_likelihood() const {
        if (std::isfinite(proposalLogAcceptanceProbability)) {
            return -model->computeNegativeLogLikelihood(this->proposal);
        } else {
            return proposalLogAcceptanceProbability;
        }
    }

    void DNest4Adapter::print(std::ostream &out) const {
        for (long i = 0; i < this->state.rows(); i++)
            out << this->state(i) << " ";
    }

    std::string DNest4Adapter::description() const {
        auto parameterNames = model->getParameterNames();
        std::string description;
        if (parameterNames) {
            for (const auto &p: parameterNames.value()) {
                description += p + " ,";
            }
            description.pop_back();
        } else {
            for (long i = 0; i < state.rows(); ++i) {
                description += "dim " + std::to_string(i) + " ,";
            }
            description.pop_back();
        }
        return description;
    }

    void DNest4Adapter::swap(DNest4Adapter &other) noexcept {
        std::swap(uniformRealDistribution, other.uniformRealDistribution);
        std::swap(state, other.state);
        std::swap(proposal, other.proposal);
        std::swap(stateLogAcceptanceProbability, other.stateLogAcceptanceProbability);
        std::swap(proposalLogAcceptanceProbability, other.proposalLogAcceptanceProbability);
        std::swap(priorProposer, other.priorProposer);
        std::swap(posteriorProposer, other.posteriorProposer);
        std::swap(model, other.model);
        std::swap(internal_rng, other.internal_rng);
    }

    DNest4Adapter &DNest4Adapter::operator=(DNest4Adapter other) {
        other.swap(*this);
        return *this;
    }

    DNest4Adapter &DNest4Adapter::operator=(DNest4Adapter &&other) noexcept {
        other.swap(*this);
        return *this;
    }

    void DNest4Adapter::accept_perturbation() {
        state = std::move(proposal);
        stateLogAcceptanceProbability = proposalLogAcceptanceProbability;
    }
}


#endif //HOPS_DNEST4ADAPTER_HPP
