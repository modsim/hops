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
    template<typename ModelImplType>
    class DNest4Adapter : public ModelImplType {
    public:
        DNest4Adapter() {

            this->state = ModelImplType::getPriorSampler()->getState();
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
        typename ModelImplType::StateType state;
        static constexpr const int numberOfPriorSteps = 1;
    };

    template<typename ModelImplType>
    void DNest4Adapter<ModelImplType>::from_prior(DNest4::RNG &rng) {
        if(!ModelImplType::isRngInitialized()) {
            int seed = rng.rand_int(std::numeric_limits<int>::max()-1);
            ModelImplType::seedRng(seed);
        }
        for (int i = 0; i < this->numberOfPriorSteps; ++i) {
            ModelImplType::getPriorSampler()->draw(ModelImplType::getRandomNumberGenerator());
        }
        this->state = ModelImplType::getPriorSampler()->getState();
        // TODO clear records
    }

    template<typename ModelImplType>
    double DNest4Adapter<ModelImplType>::perturb(DNest4::RNG &rng) {
        if(!ModelImplType::isRngInitialized()) {
            int seed = rng.rand_int(std::numeric_limits<int>::max()-1);
            ModelImplType::seedRng(seed);
        }
        // TODO check if this is correct
//        ModelImplType::getPosteriorSampler()->setState(this->state);
//        ModelImplType::getPosteriorSampler()->propose(ModelImplType::getRandomNumberGenerator());
//        double acceptanceProbability = ModelImplType::getPosteriorSampler()->computeLogAcceptanceProbability();
//        ModelImplType::getPosteriorSampler()->acceptProposal();
//        this->state = ModelImplType::getPosteriorSampler()->getState();
//        return acceptanceProbability;
        for (int i = 0; i < this->numberOfPriorSteps; ++i) {
            ModelImplType::getPriorSampler()->draw(ModelImplType::getRandomNumberGenerator());
        }
        this->state = ModelImplType::getPriorSampler()->getState();
        return 0;
    }

    template<typename ModelImplType>
    double DNest4Adapter<ModelImplType>::log_likelihood() const {
        return -ModelImplType::getPosteriorSampler()->computeNegativeLogLikelihood(this->state);
    }

    template<typename ModelImplType>
    void DNest4Adapter<ModelImplType>::print(std::ostream &out) const {
        for (long i = 0; i < ModelImplType::getPosteriorSampler()->getState().rows(); i++)
            out << ModelImplType::getPosteriorSampler()->getState()(i) << " ";
    }

    template<typename ModelImplType>
    std::string DNest4Adapter<ModelImplType>::description() const {
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
