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
         * @Brief takes state i from prior samples
         * @param rng
         */
        void from_prior(size_t i);

        /**
         * @Brief Metropolis-Hastings proposal for DNest4
         * @param rng
         * @return
         */
        double perturb(DNest4::RNG &rng);

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

        double proposalLogAcceptanceProbability = 0;

        std::unique_ptr<hops::Proposal> proposal;
        std::unique_ptr<hops::Model> model;
    };
}


#endif //HOPS_DNEST4ADAPTER_HPP
