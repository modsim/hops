#ifndef HOPS_DNEST4ADAPTER_HPP
#define HOPS_DNEST4ADAPTER_HPP

#include <dnest4/DNest4.h>

namespace hops {

    /**
     * @Brief An adapter that allows the usage of hops sampling algorithms and hops compatible models with DNest4
     * @tparam PriorSampler A chain that draws from the prior, e.g. a uniform markov chain from the MarkovChainFactory
     * @tparam PosteriorSampler A proposal mechanism for getting samples from the posterior
     * @tparam Model
     */
    template<typename PriorSampler, typename PosteriorSampler>
    class DNest4Adapter : public PriorSampler, PosteriorSampler {
    public:

        DNest4Adapter(const PriorSampler &priorSampler,
                                const PosteriorSampler &posteriorSampler):
                PriorSampler(priorSampler),
                PosteriorSampler(posteriorSampler) {
            this->state = PriorSampler::getState();
        }

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
        typename PosteriorSampler::StateType state;
        static constexpr const int numberOfPriorSteps = 100;
    };

    template<typename PriorSampler, typename PosteriorSampler>
    void DNest4Adapter<PriorSampler, PosteriorSampler>::from_prior(DNest4::RNG &rng) {
        for (int i = 0; i < this->numberOfPriorSteps; ++i) {
            PriorSampler::draw(rng);
        }
        this->state = PriorSampler::getStateRecords().back();
    }

    template<typename PriorSampler, typename PosteriorSampler>
    double DNest4Adapter<PriorSampler, PosteriorSampler>::perturb(DNest4::RNG &rng) {
        // TODO check if this is correct
        PosteriorSampler::setState(this->state);
        PosteriorSampler::propose(rng);
        PosteriorSampler::acceptProposal();
        this->state = PosteriorSampler::getState();
        return PosteriorSampler::computeLogAcceptanceChanceProbability();
    }

    template<typename PriorSampler, typename PosteriorSampler>
    double DNest4Adapter<PriorSampler, PosteriorSampler>::log_likelihood() const {
        return -PosteriorSampler::computeNegativeLogLikelihood(this->state);
    }

    template<typename PriorSampler, typename PosteriorSampler>
    void DNest4Adapter<PriorSampler, PosteriorSampler>::print(std::ostream &out) const {
        for (long i = 0; i < this->getState().get.size(); i++)
            out << this->getState(i) << " ";
    }

    template<typename PriorSampler, typename PosteriorSampler>
    std::string DNest4Adapter<PriorSampler, PosteriorSampler>::description() const {
        // TODO if implements getParameterNames() then return those as string
        std::string description;
        for (long i = 0; i < state.rows(); ++i) {
            description += "dim " + std::to_string(i) + " ,";
        }
        description.pop_back();
        return description;
    }
}


#endif //HOPS_DNEST4ADAPTER_HPP
