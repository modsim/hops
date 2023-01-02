#include "DNest4EnvironmentSingleton.hpp"

hops::DNest4EnvironmentSingleton &hops::DNest4EnvironmentSingleton::getInstance() {
    static DNest4EnvironmentSingleton instance;
    return instance;
}

[[nodiscard]] std::unique_ptr<hops::Proposal> hops::DNest4EnvironmentSingleton::getPriorProposer() const {
    return priorProposer->copyProposal();
}

[[nodiscard]] std::unique_ptr<hops::Model> hops::DNest4EnvironmentSingleton::getModel() const {
    return model->copyModel();
}

[[nodiscard]] std::unique_ptr<hops::Proposal> hops::DNest4EnvironmentSingleton::getPosteriorProposer() const {
    return posteriorProposer->copyProposal();
}

[[nodiscard]] const hops::VectorType &hops::DNest4EnvironmentSingleton::getStartingPoint() const {
    return startingPoint;
}

void hops::DNest4EnvironmentSingleton::setPriorProposer(std::unique_ptr<hops::Proposal> newProposer) {
    DNest4EnvironmentSingleton::priorProposer = std::move(newProposer);
}

void hops::DNest4EnvironmentSingleton::setModel(std::unique_ptr<hops::Model> newModel) {
    DNest4EnvironmentSingleton::model = std::move(newModel);
}

void hops::DNest4EnvironmentSingleton::setPosteriorProposer(std::unique_ptr<hops::Proposal> newPosteriorProposer) {
    DNest4EnvironmentSingleton::posteriorProposer = std::move(newPosteriorProposer);
}

void hops::DNest4EnvironmentSingleton::setStartingPoint(const VectorType &newStartingPoint) {
    DNest4EnvironmentSingleton::startingPoint = newStartingPoint;
}
