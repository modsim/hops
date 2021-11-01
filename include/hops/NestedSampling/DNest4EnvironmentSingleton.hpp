#ifndef HOPS_DNEST4ENVIRONMENTSINGLETON_HPP
#define HOPS_DNEST4ENVIRONMENTSINGLETON_HPP

#include <hops/MarkovChain/MarkovChain.hpp>

namespace hops {
    class DNest4EnvironmentSingleton {
    public:
        using StateType = Model::StateType;

        static DNest4EnvironmentSingleton &getInstance() {
            static DNest4EnvironmentSingleton instance;
            return instance;
        }

        [[nodiscard]] static bool isRngInitialized() const {
            return rngInitialized;
        }

        static hops::RandomNumberGenerator &getRandomNumberGenerator() {
            return rng;
        }

        static const hops::MarkovChain *getSampler() const {
            return sampler.get();
        }

        static const Model *getModel() const {
            return model.get();
        }

        DNest4EnvironmentSingleton(const DNest4ModelSingleton &) = delete;

        DNest4EnvironmentSingleton &operator=(const DNest4ModelSingleton &) = delete;

    private:
        static std::unique_ptr<hops::MarkovChain> sampler;
        static std::unique_ptr<hops::Model> model;

        static bool rngInitialized = false;
        static hops::RandomNumberGenerator rng;

        DNest4EnvironmentSingleton() = default;

        ~DNest4EnvironmentSingleton() = default;
    };
}

#endif //HOPS_DNEST4ENVIRONMENTSINGLETON_HPP
