#include "DNest4Adapter.hpp"

hops::DNest4Adapter::DNest4Adapter() {
    if (DNest4EnvironmentSingleton::getInstance().getProposal()) {
        proposal = DNest4EnvironmentSingleton::getInstance().getProposal()->copyProposal();
    }
    if (DNest4EnvironmentSingleton::getInstance().getModel()) {
        model = DNest4EnvironmentSingleton::getInstance().getModel()->copyModel();
    }
    proposalLogAcceptanceProbability = 0;
}

hops::DNest4Adapter::DNest4Adapter(const DNest4Adapter &other) {
    proposalLogAcceptanceProbability = other.proposalLogAcceptanceProbability;
    if (other.proposal) {
        proposal = other.proposal->copyProposal();
    }
    if (other.model) {
        model = other.model->copyModel();
    }
}

hops::DNest4Adapter::DNest4Adapter(DNest4Adapter &&other) noexcept {
    proposalLogAcceptanceProbability = other.proposalLogAcceptanceProbability;
    proposal = std::move(other.proposal);
    model = std::move(other.model);
}

void hops::DNest4Adapter::from_prior(size_t i) {
    proposal->setState(DNest4EnvironmentSingleton::getInstance().getPriorSample(i));
}

double hops::DNest4Adapter::perturb(DNest4::RNG &rng) {
    proposal->propose(rng.engine);
    // In case the proposer uses the loglikelihoods directly, we subtract them again, because DNEST4
    // expects this acceptance chance to not contain them already.
    // If the proposers doesn't know the loglikelihoods, they are 0 anyways.
    proposalLogAcceptanceProbability = proposal->computeLogAcceptanceProbability()
                                       + proposal->getProposalNegativeLogLikelihood()
                                       - proposal->getStateNegativeLogLikelihood();
    return proposalLogAcceptanceProbability;
}

double hops::DNest4Adapter::log_likelihood() const {
    return -model->computeNegativeLogLikelihood(this->proposal->getState());
}

double hops::DNest4Adapter::proposal_log_likelihood() const {
    // Proposals are not always valid states, this is reflected in the proposalLogAcceptanceProbability
    if (std::isfinite(proposalLogAcceptanceProbability)) {
        return -model->computeNegativeLogLikelihood(this->proposal->getProposal());
    }
    return -std::numeric_limits<double>::infinity();
}

void hops::DNest4Adapter::print(std::ostream &out) const {
    for (long i = 0; i < proposal->getState().rows(); i++)
        out << this->proposal->getState()(i) << " ";
}

std::string hops::DNest4Adapter::description() const {
    std::string description;
    auto parameterNames = proposal->getDimensionNames();
    for (const auto &p: parameterNames) {
        description += p + " ,";
    }
    description.pop_back();
    return description;
}

void hops::DNest4Adapter::swap(DNest4Adapter &other) noexcept {
    std::swap(proposalLogAcceptanceProbability, other.proposalLogAcceptanceProbability);
    std::swap(proposal, other.proposal);
    std::swap(model, other.model);
}

hops::DNest4Adapter &hops::DNest4Adapter::operator=(DNest4Adapter other) {
    other.swap(*this);
    return *this;
}

hops::DNest4Adapter &hops::DNest4Adapter::operator=(DNest4Adapter &&other) noexcept {
    other.swap(*this);
    return *this;
}

void hops::DNest4Adapter::accept_perturbation() {
    proposal->acceptProposal();
    // after acceptProposal accepting proposal again is not valid.
    proposalLogAcceptanceProbability = -std::numeric_limits<double>::infinity();
}
