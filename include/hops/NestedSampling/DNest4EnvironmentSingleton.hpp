#ifndef HOPS_DNEST4ENVIRONMENTSINGLETON_HPP
#define HOPS_DNEST4ENVIRONMENTSINGLETON_HPP

#include <hops/MarkovChain/Proposal/Proposal.hpp>
#include <hops/Model/Model.hpp>

namespace hops {
    class DNest4EnvironmentSingleton {
    public:
        static DNest4EnvironmentSingleton &getInstance() {
            static DNest4EnvironmentSingleton instance;
            return instance;
        }

        [[nodiscard]] bool isRngInitialized() const {
            return rngInitialized;
        }

        void initializeRng(int seed, int stream=0) {
            rngInitialized = true;
            rng = RandomNumberGenerator(seed, stream);
        }

        hops::RandomNumberGenerator &getRandomNumberGenerator() {
            return rng;
        }

        [[nodiscard]] std::shared_ptr<hops::Proposal> getProposer() const {
            return proposer;
        }

        [[nodiscard]] std::shared_ptr<Model> getModel() const {
            return model;
        }

        void setProposer(const std::shared_ptr<hops::Proposal> &newProposer) {
            DNest4EnvironmentSingleton::proposer = newProposer;
        }

        void setModel(const std::shared_ptr<hops::Model> &newModel) {
            DNest4EnvironmentSingleton::model = newModel;
        }

        DNest4EnvironmentSingleton(const DNest4EnvironmentSingleton &) = delete;

        DNest4EnvironmentSingleton &operator=(const DNest4EnvironmentSingleton &) = delete;

    private:
        std::shared_ptr<hops::Proposal> proposer;
        std::shared_ptr<hops::Model> model;

        bool rngInitialized = false;
        hops::RandomNumberGenerator rng;

        DNest4EnvironmentSingleton() = default;

        ~DNest4EnvironmentSingleton() = default;
    };
}

#endif //HOPS_DNEST4ENVIRONMENTSINGLETON_HPP
