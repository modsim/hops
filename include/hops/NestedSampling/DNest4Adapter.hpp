#ifndef HOPS_DNEST4ADAPTER_HPP
#define HOPS_DNEST4ADAPTER_HPP

#include <dnest4/DNest4.h>

namespace hops {

    /**
     * @Brief An adapter that allows the usage of hops sampling algorithms and hops compatible models with DNest4
     */
    template<typename Environment>
    class DNest4Adapter{
    public:
        DNest4Adapter() {
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
        Environment environment;
        typename Environment::StateType state;
    };

    template<typename ModelImplType>
    void DNest4Adapter<ModelImplType>::from_prior(DNest4::RNG &rng) {
        if(!environment.isRngInitialized()) {
            int seed = rng.rand_int(std::numeric_limits<int>::max()-1);
            environment.seedRng(seed);
        }
        ModelImplType::getSampler()->draw(ModelImplType::getRandomNumberGenerator());
        this->state = ModelImplType::getSampler()->getState();
    }

    template<typename ModelImplType>
    double DNest4Adapter<ModelImplType>::perturb(DNest4::RNG &rng) {
        if(!ModelImplType::isRngInitialized()) {
            int seed = rng.rand_int(std::numeric_limits<int>::max()-1);
            ModelImplType::seedRng(seed);
        }
        ModelImplType::getSampler()->draw(ModelImplType::getRandomNumberGenerator());
        this->state = ModelImplType::getSampler()->getState();
        return ModelImplType::getSampler()->computeLogAcceptanceProbability();
    }

    template<typename ModelImplType>
    double DNest4Adapter<ModelImplType>::log_likelihood() const {
        return -ModelImplType::getModel()->computeNegativeLogLikelihood(this->state);
    }

    template<typename ModelImplType>
    void DNest4Adapter<ModelImplType>::print(std::ostream &out) const {
        for (long i = 0; i < this->state.rows(); i++)
            out << this->state(i) << " ";
    }

    template<typename ModelImplType>
    std::string DNest4Adapter<ModelImplType>::description() const {
         TODO if implements getParameterNames() then return those as string
         model->getPara
        std::string description;
        for (long i = 0; i < state.rows(); ++i) {
            description += "dim " + std::to_string(i) + " ,";
        }
        description.pop_back();
        return description;

                environment = Environment::getInstance();
        this->state = environment.getSampler()->getState();
    }
}


#endif //HOPS_DNEST4ADAPTER_HPP
