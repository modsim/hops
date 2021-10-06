#ifndef HOPS_DNEST4ADAPTER_HPP
#define HOPS_DNEST4ADAPTER_HPP

#include <dnest4/DNest4.h>

namespace hops {

    /**
     * @Brief DNest4 requires ModelTypes to be default constructable. This helper class allows making the
     *          DNest4Adapter default-constructable while still passing data about the polytope and the samplers.
     * @tparam PriorSampler
     * @tparam PosteriorSampler
     */
    template<typename PriorSampler, typename PosteriorSampler>
    class DNest4AdapterConstructor {
    public:
        DNest4AdapterConstructor(PriorSampler priorSampler, PosteriorSampler posteriorSampler) :
                priorSampler(std::move(priorSampler)),
                posteriorSampler(std::move(posteriorSampler)) {}

        PriorSampler getPriorSampler() const {
            return priorSampler;
        }

        PosteriorSampler getPosteriorSampler() const {
            return posteriorSampler;
        }

    private:
        PriorSampler priorSampler;
        PosteriorSampler posteriorSampler;
    };

    /**
     * @Brief An adapter that allows the usage of hops sampling algorithms and hops compatible models with DNest4
     * @tparam PriorSampler A chain that draws from the prior, e.g. a uniform markov chain from the MarkovChainFactory
     * @tparam PosteriorSampler A proposal mechanism for getting samples from the posterior
     * @tparam Model
     */
    template<typename PriorSampler, typename PosteriorSampler, DNest4AdapterConstructor<PriorSampler, PosteriorSampler> *constructor>
    class DNest4Adapter {
    public:

        DNest4Adapter() {
            priorSampler = constructor->getPriorSampler();
            posteriorSampler = constructor->getPosteriorSampler();
            this->state = priorSampler.getState();
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

        PriorSampler priorSampler;
        PosteriorSampler posteriorSampler;
    };

    template<typename PriorSampler, typename PosteriorSampler, DNest4AdapterConstructor<PriorSampler, PosteriorSampler> * constructor>
    void DNest4Adapter<PriorSampler, PosteriorSampler, constructor>::from_prior(DNest4::RNG &rng) {
//        for (int i = 0; i < this->numberOfPriorSteps; ++i) {
//            PriorSampler::draw(rng);
//        }
//        this->state = PriorSampler::getStateRecords().back();
    }

    template<typename PriorSampler, typename PosteriorSampler, DNest4AdapterConstructor<PriorSampler, PosteriorSampler> * constructor>
    double DNest4Adapter<PriorSampler, PosteriorSampler, constructor>::perturb(DNest4::RNG &rng) {
        // TODO check if this is correct
//        PosteriorSampler::setState(this->state);
//        PosteriorSampler::propose(rng);
//        PosteriorSampler::acceptProposal();
//        this->state = PosteriorSampler::getState();
//        return PosteriorSampler::computeLogAcceptanceChanceProbability();
    }

    template<typename PriorSampler, typename PosteriorSampler, DNest4AdapterConstructor<PriorSampler, PosteriorSampler> * constructor>
    double DNest4Adapter<PriorSampler, PosteriorSampler, constructor>::log_likelihood() const {
//        return -PosteriorSampler::computeNegativeLogLikelihood(this->state);
    }

    template<typename PriorSampler, typename PosteriorSampler, DNest4AdapterConstructor<PriorSampler, PosteriorSampler> * constructor>
    void DNest4Adapter<PriorSampler, PosteriorSampler, constructor>::print(std::ostream &out) const {
        for (long i = 0; i < this->getState().get.size(); i++)
            out << this->getState(i) << " ";
    }

    template<typename PriorSampler, typename PosteriorSampler, DNest4AdapterConstructor<PriorSampler, PosteriorSampler> * constructor>
    std::string DNest4Adapter<PriorSampler, PosteriorSampler, constructor>::description() const {
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
