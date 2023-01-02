#ifndef HOPS_DNEST4ENVIRONMENTSINGLETON_HPP
#define HOPS_DNEST4ENVIRONMENTSINGLETON_HPP

#include "hops/MarkovChain/Proposal/Proposal.hpp"
#include "hops/Model/Model.hpp"

namespace hops {
    class DNest4EnvironmentSingleton {
    public:
        static DNest4EnvironmentSingleton &getInstance();

        [[nodiscard]] std::unique_ptr<hops::Proposal> getPriorProposer() const;

        [[nodiscard]] std::unique_ptr<Model> getModel() const;

        [[nodiscard]] std::unique_ptr<hops::Proposal> getPosteriorProposer() const;

        [[nodiscard]] const VectorType &getStartingPoint() const;

        /**
         * @brief The prior proposer should be maximally efficient in proposing states from the uniform prior
         * (e.g. a proposer based on CHRR).
         */
        void setPriorProposer(std::unique_ptr<hops::Proposal> newProposer);

        void setModel(std::unique_ptr<hops::Model> newModel);

        /**
         * @brief The posterior proposer should be geared towards efficiently sampling the posterior distribution
         */
        void setPosteriorProposer(std::unique_ptr<hops::Proposal> newPosteriorProposer);

        void setStartingPoint(const VectorType &newStartingPoint);

        DNest4EnvironmentSingleton(const DNest4EnvironmentSingleton &) = delete;

        DNest4EnvironmentSingleton &operator=(const DNest4EnvironmentSingleton &) = delete;

    private:
        std::unique_ptr<hops::Proposal> priorProposer;
        std::unique_ptr<hops::Proposal> posteriorProposer;
        std::unique_ptr<hops::Model> model;
        VectorType startingPoint;

        DNest4EnvironmentSingleton() = default;

        ~DNest4EnvironmentSingleton() = default;
    };
}

#endif //HOPS_DNEST4ENVIRONMENTSINGLETON_HPP
