#ifndef HOPS_DNEST4ADAPTER_HPP
#define HOPS_DNEST4ADAPTER_HPP

#include <dnest4/DNest4.h>
#include <hops/MarkovChain/MarkovChain.hpp>

#include "DNest4EnvironmentSingleton.hpp"

namespace hops {

    /**
     * @Brief An adapter that allows the usage of hops sampling algorithms and hops compatible models with DNest4
     */
    class DNest4Adapter {
    public:
        DNest4Adapter();

        DNest4Adapter(const DNest4Adapter &other);

        DNest4Adapter(DNest4Adapter &&other) noexcept;

        DNest4Adapter &operator=(DNest4Adapter other);

        /**
         * @Brief generates a state from the prior for DNest4
         * @param rng
         */
        void from_prior(DNest4::RNG &);

        /**
         * @ Metropolis-Hastings proposal for DNest4
         * @param rng
         * @return
         */
        double perturb(DNest4::RNG &);

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
        void swap(DNest4Adapter &other) noexcept;

        std::uniform_real_distribution<double> uniformRealDistribution;

        VectorType state;
        double stateLogAcceptanceProbability = 0;

        std::unique_ptr<hops::Proposal> priorProposer;
        std::unique_ptr<hops::Proposal> posteriorProposer;
        std::unique_ptr<hops::Model> model;

        RandomNumberGenerator internal_rng;
    };

    DNest4Adapter::DNest4Adapter() {
        priorProposer = DNest4EnvironmentSingleton::getInstance().getPriorProposer()->deepCopy();
        posteriorProposer = DNest4EnvironmentSingleton::getInstance().getPosteriorProposer()->deepCopy();
        model = DNest4EnvironmentSingleton::getInstance().getModel()->deepCopy();

        std::random_device seedDevice;
        std::uniform_int_distribution<long> dist(std::numeric_limits<long>::min(),
                                                 std::numeric_limits<long>::max());
        long seed = dist(seedDevice);
        internal_rng.seed(seed);

        state = DNest4EnvironmentSingleton::getInstance().getStartingPoint();
        stateLogAcceptanceProbability = 0;
    }

    DNest4Adapter::DNest4Adapter(const DNest4Adapter &other) {
        uniformRealDistribution = other.uniformRealDistribution;
        state = other.state;
        stateLogAcceptanceProbability = other.stateLogAcceptanceProbability;
        priorProposer = other.priorProposer->deepCopy();
        posteriorProposer = other.posteriorProposer->deepCopy();
        model = other.model->deepCopy();
        internal_rng = other.internal_rng;
    }

    DNest4Adapter::DNest4Adapter(DNest4Adapter &&other) noexcept {
        uniformRealDistribution = other.uniformRealDistribution;
        state = std::move(other.state);
        stateLogAcceptanceProbability = other.stateLogAcceptanceProbability;
        priorProposer = std::move(other.priorProposer);
        posteriorProposer = std::move(other.posteriorProposer);
        model = std::move(other.model);
        internal_rng = other.internal_rng;
    }

    void DNest4Adapter::from_prior(DNest4::RNG &) {
        for (int i = 0; i < 100; ++i) {
            auto[logAcceptanceProbability, proposal] = priorProposer->propose(internal_rng);
            double logAcceptanceChance = std::log(uniformRealDistribution(internal_rng));
            if (logAcceptanceChance < logAcceptanceProbability) {
                this->state = priorProposer->acceptProposal();
            }
        }
    }

    double DNest4Adapter::perturb(DNest4::RNG &) {
        for (int i = 0; i < 100; ++i) {
            auto[logAcceptanceProbability, proposal] = priorProposer->propose(internal_rng);
            double logAcceptanceChance = std::log(uniformRealDistribution(internal_rng));
            if (logAcceptanceChance < logAcceptanceProbability) {
                this->state = priorProposer->acceptProposal();
            }
        }
        return 0;
    }

    double DNest4Adapter::log_likelihood() const {
        return -model->computeNegativeLogLikelihood(this->state);
    }

    void DNest4Adapter::print(std::ostream &out) const {
        for (long i = 0; i < this->state.rows(); i++)
            out << this->state(i) << " ";
    }

    std::string DNest4Adapter::description() const {
        auto parameterNames = model->getParameterNames();
        std::string description;
        if (parameterNames) {
            for (const auto &p: parameterNames.value()) {
                description += p + " ,";
            }
            description.pop_back();
        } else {
            for (long i = 0; i < state.rows(); ++i) {
                description += "dim " + std::to_string(i) + " ,";
            }
            description.pop_back();
        }
        return description;
    }

    void DNest4Adapter::swap(DNest4Adapter &other) noexcept {
        std::swap(uniformRealDistribution, other.uniformRealDistribution);
        std::swap(state, other.state);
        std::swap(stateLogAcceptanceProbability, other.stateLogAcceptanceProbability);
        std::swap(priorProposer, other.priorProposer);
        std::swap(posteriorProposer, other.posteriorProposer);
        std::swap(model, other.model);
        std::swap(internal_rng, other.internal_rng);
    }

    DNest4Adapter &DNest4Adapter::operator=(DNest4Adapter other) {
        other.swap(*this);
        return *this;
    }
}


#endif //HOPS_DNEST4ADAPTER_HPP
