#ifndef HOPS_DNEST4ADAPTER_HPP
#define HOPS_DNEST4ADAPTER_HPP

#include "hops/extern/DNest4.hpp"
#include "hops/MarkovChain/MarkovChain.hpp"

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
}


#endif //HOPS_DNEST4ADAPTER_HPP
