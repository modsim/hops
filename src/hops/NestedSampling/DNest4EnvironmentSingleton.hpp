#ifndef HOPS_DNEST4ENVIRONMENTSINGLETON_HPP
#define HOPS_DNEST4ENVIRONMENTSINGLETON_HPP

#include "hops/MarkovChain/Proposal/Proposal.hpp"
#include "hops/Model/Model.hpp"

namespace hops {
    class DNest4EnvironmentSingleton {
    public:
        static DNest4EnvironmentSingleton &getInstance();

        [[nodiscard]] std::unique_ptr<Model> getModel() const;

        [[nodiscard]] std::unique_ptr<Proposal> getProposal() const;

        VectorType getPriorSample(size_t i);

        void setPriorSamples(std::vector<VectorType> prior_samples);

        void setModel(std::unique_ptr<hops::Model> newModel);

        /**
         * @brief The posterior proposer should be geared towards efficiently sampling the posterior distribution
         */
        void setProposal(std::unique_ptr<hops::Proposal> newProposal);

        DNest4EnvironmentSingleton(const DNest4EnvironmentSingleton &) = delete;

        DNest4EnvironmentSingleton &operator=(const DNest4EnvironmentSingleton &) = delete;

    private:
        std::vector<VectorType> prior_samples;
        std::unique_ptr<hops::Proposal> proposal;
        std::unique_ptr<hops::Model> model;
        size_t numberOfPriorSteps; // number of steps for the priorProposer to take until we take on sample of it as initial value.

        DNest4EnvironmentSingleton() = default;

        ~DNest4EnvironmentSingleton() = default;
    };
}

#endif //HOPS_DNEST4ENVIRONMENTSINGLETON_HPP
