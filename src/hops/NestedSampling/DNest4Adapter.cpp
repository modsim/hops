#include "DNest4Adapter.hpp"

hops::DNest4Adapter::DNest4Adapter() {
    priorProposer = DNest4EnvironmentSingleton::getInstance().getPriorProposer();
    posteriorProposer = DNest4EnvironmentSingleton::getInstance().getPosteriorProposer();
    model = DNest4EnvironmentSingleton::getInstance().getModel();

    std::random_device seedDevice;
    std::uniform_int_distribution<long> dist(std::numeric_limits<long>::min(),
                                             std::numeric_limits<long>::max());
    long seed = dist(seedDevice);
    internal_rng.seed(seed);

    state = DNest4EnvironmentSingleton::getInstance().getStartingPoint();
    proposal = state;
    stateLogAcceptanceProbability = 0;
    proposalLogAcceptanceProbability = 0;

}

hops::DNest4Adapter::DNest4Adapter(const DNest4Adapter &other) {
    uniformRealDistribution = other.uniformRealDistribution;
    state = other.state;
    proposal = other.proposal;
    stateLogAcceptanceProbability = other.stateLogAcceptanceProbability;
    proposalLogAcceptanceProbability = other.proposalLogAcceptanceProbability;
    priorProposer = other.priorProposer->copyProposal();
    posteriorProposer = other.posteriorProposer->copyProposal();
    model = other.model->copyModel();
    internal_rng = other.internal_rng;
}

hops::DNest4Adapter::DNest4Adapter(DNest4Adapter &&other) noexcept {
    uniformRealDistribution = other.uniformRealDistribution;
    state = std::move(other.state);
    proposal = std::move(other.proposal);
    stateLogAcceptanceProbability = other.stateLogAcceptanceProbability;
    proposalLogAcceptanceProbability = other.proposalLogAcceptanceProbability;
    priorProposer = std::move(other.priorProposer);
    posteriorProposer = std::move(other.posteriorProposer);
    model = std::move(other.model);
    internal_rng = other.internal_rng;
}

void hops::DNest4Adapter::from_prior(DNest4::RNG &) {
    priorProposer->propose(internal_rng);
    // In case the proposer uses the loglikelihoods directly, we subtract them again, because DNEST4
    // expects this  acceptance chance to not contain them already.
    // If the proposers doesn't know the loglikelihoods, they are 0 anyways.
    double logAcceptanceProbability = priorProposer->computeLogAcceptanceProbability()
                                      + priorProposer->getProposalNegativeLogLikelihood()
                                      - priorProposer->getStateNegativeLogLikelihood();
    double logAcceptanceChance = std::log(uniformRealDistribution(internal_rng));
    if (logAcceptanceChance < logAcceptanceProbability && std::isfinite(logAcceptanceProbability)) {
        this->state = priorProposer->acceptProposal();
    }
}

double hops::DNest4Adapter::perturb(DNest4::RNG &) {
    posteriorProposer->setState(state);
    proposal = posteriorProposer->propose(internal_rng);
    // In case the proposer uses the loglikelihoods directly, we subtract them again, because DNEST4
    // expects this  acceptance chance to not contain them already.
    // If the proposers doesn't know the loglikelihoods, they are 0 anyways.
    proposalLogAcceptanceProbability = posteriorProposer->computeLogAcceptanceProbability()
                                       + posteriorProposer->getProposalNegativeLogLikelihood()
                                       - posteriorProposer->getStateNegativeLogLikelihood();
    return proposalLogAcceptanceProbability;
}

double hops::DNest4Adapter::log_likelihood() const {
    if (std::isfinite(stateLogAcceptanceProbability)) {
        return -model->computeNegativeLogLikelihood(this->state);
    }
    return -std::numeric_limits<double>::infinity();
}

double hops::DNest4Adapter::proposal_log_likelihood() const {
    if (std::isfinite(proposalLogAcceptanceProbability)) {
        return -model->computeNegativeLogLikelihood(this->proposal);
    }
    return -std::numeric_limits<double>::infinity();
}

void hops::DNest4Adapter::print(std::ostream &out) const {
    for (long i = 0; i < this->state.rows(); i++)
        out << this->state(i) << " ";
}

std::string hops::DNest4Adapter::description() const {
    std::string description;
    auto parameterNames = posteriorProposer->getDimensionNames();
    for (const auto &p: parameterNames) {
        description += p + " ,";
    }
    description.pop_back();
    return description;
}

void hops::DNest4Adapter::swap(DNest4Adapter &other) noexcept {
    std::swap(uniformRealDistribution, other.uniformRealDistribution);
    std::swap(state, other.state);
    std::swap(proposal, other.proposal);
    std::swap(stateLogAcceptanceProbability, other.stateLogAcceptanceProbability);
    std::swap(proposalLogAcceptanceProbability, other.proposalLogAcceptanceProbability);
    std::swap(priorProposer, other.priorProposer);
    std::swap(posteriorProposer, other.posteriorProposer);
    std::swap(model, other.model);
    std::swap(internal_rng, other.internal_rng);
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
    state = proposal;
    stateLogAcceptanceProbability = proposalLogAcceptanceProbability;
}
