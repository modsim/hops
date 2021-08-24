#ifndef HOPS_DNEST4ADAPTER_HPP
#define HOPS_DNEST4ADAPTER_HPP

#include <dnest4/DNest4.h>

namespace hops {

    /**
     * @Brief An adapter that allows the usage of hops sampling algorithms and hops compatible models with DNest4
     * @tparam PriorSampler A chain that draws from the prior, e.g. a uniform markov chain from the MarkovChainFactory
     * @tparam PosteriorProposer A proposal mechanism for getting samples from the posterior
     * @tparam Model
     */
    template<typename PriorSampler, typename PosteriorProposer, typename Model>
    class DNest4DAdapter {
    public:
        explicit DNest4DAdapter(const PriorSampler &priorSampler,
                                const PosteriorProposer &PosteriorProposer,
                                const Model &model) :
                PriorSampler(priorSampler),
                PosteriorProposer(PosteriorProposer),
                Model(model) {
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
        MarkovChain::StateType state;
        static constexpr const int numberOfPriorSteps = 100;
    };

    template<typename PriorMarkovChain, typename PosteriorProposer, typename Model>
    void DNest4DAdapter<PriorMarkovChain, PosteriorProposer, Model>::from_prior(DNest4::RNG &rng) {
        for (int i = 0; i < this->numberOfPriorSteps; ++i) {
            PriorMarkovChain::draw(rng);
        }
        this->state = PriorMarkovChain::getStateRecords().back();
    }

    template<typename PriorMarkovChain, typename PosteriorProposer, typename Model>
    double DNest4DAdapter<PriorMarkovChain, PosteriorProposer, Model>::perturb(DNest4::RNG &rng) {
        PosteriorProposer.propose(rng);
        PosteriorProposer.acceptProposal();
        this->state = PosteriorProposer.getState();
        return PosteriorProposer.computeLogAcceptanceChanceProbability();
    }

    template<typename PriorMarkovChain, typename PosteriorProposer, typename Model>
    double DNest4DAdapter<PriorMarkovChain, PosteriorProposer, Model>::log_likelihood() const {
        return -model.computeNegativeLogLikelihood(this->state);
    }

    template<typename PriorMarkovChain, typename PosteriorProposer, typename Model>
    void DNest4DAdapter<PriorMarkovChain, PosteriorProposer, Model>::print(std::ostream &out) const {
        for (long i = 0; i < this->getState().get.size(); i++)
            out << this->getState(i) << ' ';
    }

    template<typename PriorMarkovChain, typename PosteriorProposer, typename Model>
    std::string DNest4DAdapter<PriorMarkovChain, PosteriorProposer, Model>::description() const {
        // TODO if implements getParameterNames() then return those as string
        std::string description;
        for (long i = 0; i < state.rows(); ++i) {
            description += "dim " + std::to_string(i) + " ,"
        }
        description.pop_back();
        return description;
    }

}


#endif //HOPS_DNEST4ADAPTER_HPP
