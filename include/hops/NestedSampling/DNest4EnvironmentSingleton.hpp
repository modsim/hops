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

        [[nodiscard]] std::shared_ptr<hops::Proposal> getPriorProposer() const {
            return priorProposer;
        }

        [[nodiscard]] std::shared_ptr<Model> getModel() const {
            return model;
        }

        [[nodiscard]] std::shared_ptr<hops::Proposal> getPosteriorProposer() const {
            return posteriorProposer;
        }

        [[nodiscard]] const VectorType &getStartingPoint() const {
            return startingPoint;
        }

        /**
         * @brief The prior proposer should be maximally efficient in proposing states from the uniform prior
         * (e.g. a proposer based on CHRR).
         */
        void setPriorProposer(const std::shared_ptr<hops::Proposal> &newProposer) {
            DNest4EnvironmentSingleton::priorProposer = newProposer;
        }

        void setModel(const std::shared_ptr<hops::Model> &newModel) {
            DNest4EnvironmentSingleton::model = newModel;
        }

        /**
         * @brief The posterior proposer should be geared towards efficiently sampling the posterior distribution
         */
        void setPosteriorProposer(const std::shared_ptr<hops::Proposal> &newPosteriorProposer) {
            DNest4EnvironmentSingleton::posteriorProposer = newPosteriorProposer;
        }

        void setStartingPoint(const VectorType &newStartingPoint) {
            DNest4EnvironmentSingleton::startingPoint = newStartingPoint;
        }

        DNest4EnvironmentSingleton(const DNest4EnvironmentSingleton &) = delete;

        DNest4EnvironmentSingleton &operator=(const DNest4EnvironmentSingleton &) = delete;

    private:
        std::shared_ptr<hops::Proposal> priorProposer;
        std::shared_ptr<hops::Proposal> posteriorProposer;
        std::shared_ptr<hops::Model> model;
        VectorType startingPoint;

        DNest4EnvironmentSingleton() = default;
        ~DNest4EnvironmentSingleton() = default;
    };
}

#endif //HOPS_DNEST4ENVIRONMENTSINGLETON_HPP
