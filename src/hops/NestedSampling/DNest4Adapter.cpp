#include "DNest4Adapter.hpp"
#include "hops/MarkovChain/Proposal/ReversibleJumpProposal.hpp"


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
    for (long i = 0; i < this->proposal->getState().rows(); i++) {
        out << this->proposal->getState()(i) << ' ';
    }
}

void hops::DNest4Adapter::print_internal(std::ostream &out) const {
    for (long i = 0; i < this->proposal->getProposal().rows(); i++) {
        out << this->proposal->getProposal()(i) << ' ';
    }
    auto ptr = dynamic_cast<ReversibleJumpProposal*>(this->proposal.get());
    if(ptr != nullptr) {
        out << ptr->isLastProposalJumpedModel() << ' ';
    }
}

void hops::DNest4Adapter::read(std::istream &in) {
    hops::VectorType readState = this->proposal->getState();
    std::string temp_str;
    for (long i = 0; i < readState.rows(); i++) {
        in >> temp_str;
        readState(i) = strtod(temp_str.c_str(), NULL);
    }
    this->proposal->setState(readState);
}

void hops::DNest4Adapter::read_internal(std::istream &in) {
    hops::VectorType readProposal = this->proposal->getProposal();
    std::string temp_str;
    for (long i = 0; i < readProposal.rows(); i++) {
        in >> temp_str;
        readProposal(i) = strtod(temp_str.c_str(), NULL);
    }
    this->proposal->setProposal(readProposal);
    auto ptr = dynamic_cast<ReversibleJumpProposal*>(this->proposal.get());
    if(ptr != nullptr) {
        bool isLastProposalJumpedModel;
        in >> isLastProposalJumpedModel;
        ptr->setLastProposalJumpedModel(isLastProposalJumpedModel);
    }
}

std::string hops::DNest4Adapter::description() const {
    std::string description;
    auto parameterNames = proposal->getDimensionNames();
    for (const auto &p: parameterNames) {
        description += p + ", ";
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
