#include "DNest4EnvironmentSingleton.hpp"

hops::DNest4EnvironmentSingleton &hops::DNest4EnvironmentSingleton::getInstance() {
    static DNest4EnvironmentSingleton instance;
    return instance;
}

[[nodiscard]] std::unique_ptr<hops::Model> hops::DNest4EnvironmentSingleton::getModel() const {
    if (!model) {
        return nullptr;
    }
    return model->copyModel();
}

[[nodiscard]] std::unique_ptr<hops::Proposal> hops::DNest4EnvironmentSingleton::getProposal() const {
    if (!proposal) {
        return nullptr;
    }
    return proposal->copyProposal();
}

void hops::DNest4EnvironmentSingleton::setModel(std::unique_ptr<hops::Model> newModel) {
    DNest4EnvironmentSingleton::model = std::move(newModel);
}

void hops::DNest4EnvironmentSingleton::setProposal(std::unique_ptr<hops::Proposal> newProposal) {
    DNest4EnvironmentSingleton::proposal = std::move(newProposal);
}

hops::VectorType hops::DNest4EnvironmentSingleton::getPriorSample(size_t i) {
    return prior_samples.at(i);
}

void hops::DNest4EnvironmentSingleton::setPriorSamples(std::vector<VectorType> new_prior_samples) {
    prior_samples = std::move(new_prior_samples);
}
