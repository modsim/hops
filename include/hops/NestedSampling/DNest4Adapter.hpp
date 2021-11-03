#ifndef HOPS_DNEST4ADAPTER_HPP
#define HOPS_DNEST4ADAPTER_HPP

#include <dnest4/DNest4.h>
#include <hops/MarkovChain/MarkovChain.hpp>

#include "DNest4EnvironmentSingleton.hpp"

namespace hops {

    /**
     * @Brief An adapter that allows the usage of hops sampling algorithms and hops compatible models with DNest4
     */
    class DNest4Adapter {
    public:
        DNest4Adapter() = default;

        /**
         * @Brief generates a state from the prior for DNest4
         * @param rng
         */
        void from_prior(DNest4::RNG &rng);

        /**
         * @ Metropolis-Hastings proposal for DNest4
         * @param rng
         * @return
         */
        double perturb(DNest4::RNG &rng);

        [[nodiscard]] double log_likelihood() const;

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
        static void checkAndInitializeRNG(DNest4::RNG& externalRng);

        std::uniform_real_distribution<double> uniformRealDistribution;

        VectorType state;
    };

    void DNest4Adapter::from_prior(DNest4::RNG &rng) {
        hops::DNest4Adapter::checkAndInitializeRNG(rng);
        hops::RandomNumberGenerator& internal_rng = DNest4EnvironmentSingleton::getInstance().getRandomNumberGenerator();
        std::shared_ptr<hops::Proposal> proposer = DNest4EnvironmentSingleton::getInstance().getProposer();
        auto [logAcceptanceProbability, proposal] = proposer->propose(internal_rng);
        double logAcceptanceChance = std::log(uniformRealDistribution(internal_rng));
        if (logAcceptanceChance < logAcceptanceProbability) {
            this->state = proposal;
        }
    }

    double DNest4Adapter::perturb(DNest4::RNG &rng) {
        hops::DNest4Adapter::checkAndInitializeRNG(rng);
        hops::RandomNumberGenerator& internal_rng = DNest4EnvironmentSingleton::getInstance().getRandomNumberGenerator();
        std::shared_ptr<hops::Proposal> proposer = DNest4EnvironmentSingleton::getInstance().getProposer();
        auto [logAcceptanceProbability, proposal] = proposer->propose(internal_rng);
        if(std::isfinite(logAcceptanceProbability)) {
            // If logAcceptanceProbability is finite do the internal setting of new state
            this->state = proposer->acceptProposal();
        }
        else {
            // If logAcceptanceProbability is not finite, do not set state internally, only set it here for DNest4
            this->state = proposal;
        }
        return logAcceptanceProbability;
    }

    double DNest4Adapter::log_likelihood() const {
        return -DNest4EnvironmentSingleton::getInstance().getModel()->computeNegativeLogLikelihood(this->state);
    }

    void DNest4Adapter::print(std::ostream &out) const {
        for (long i = 0; i < this->state.rows(); i++)
            out << this->state(i) << " ";
    }

    std::string DNest4Adapter::description() const {
        auto parameterNames = DNest4EnvironmentSingleton::getInstance().getModel()->getParameterNames();
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

    void DNest4Adapter::checkAndInitializeRNG(DNest4::RNG &externalRng) {
        if (!DNest4EnvironmentSingleton::getInstance().isRngInitialized()) {
            int seed = externalRng.rand_int(std::numeric_limits<int>::max() - 1);
            DNest4EnvironmentSingleton::getInstance().initializeRng(seed);
        }
    }
}


#endif //HOPS_DNEST4ADAPTER_HPP
