#include "ReversibleJumpProposal.hpp"
#include "hops/Utility/StringUtility.hpp"

namespace {
    std::pair<double, double>
    distanceInCoordinateDirection(const Eigen::MatrixXd &A, const Eigen::VectorXd &b, double currentValue,
                                  long coordinate) {
        Eigen::VectorXd slacks = b - A.col(coordinate) * currentValue;
        Eigen::VectorXd inverseDistances = A.col(coordinate).cwiseQuotient(slacks);
        // Inverse distance are potentially nan due to default values on the boundary of the polytope.
        // Replaces nan because nan should not influence the distances.
        inverseDistances = inverseDistances
                .array()
                .unaryExpr([](double value) { return std::isnan(value) ? 0. : value; })
                .matrix();
        double forwardDistance = 1. / inverseDistances.maxCoeff();
        double backwardDistance = 1. / inverseDistances.minCoeff();
        forwardDistance = (forwardDistance < 0) ? 0 : forwardDistance;
        backwardDistance = (backwardDistance > 0) ? 0 : backwardDistance;
        assert(backwardDistance <= 0 && forwardDistance >= 0);
        return std::make_pair(backwardDistance, forwardDistance);
    }
}


hops::ReversibleJumpProposal::ReversibleJumpProposal(std::unique_ptr<Proposal> proposalImpl,
                                                     const Eigen::VectorXi &jumpIndices,
                                                     const VectorType &parameterDefaultValues,
                                                     const std::optional<Eigen::MatrixXd> &A,
                                                     const std::optional<Eigen::VectorXd> &b) :
        proposalImpl(std::move(proposalImpl)),
        jumpIndices(jumpIndices),
        defaultValues(parameterDefaultValues) {

    if (this->jumpIndices.rows() != this->defaultValues.rows()) {
        throw std::runtime_error("dimension missmatch in input");
    }
    if (!this->proposalImpl) {
        throw std::runtime_error("proposal mechanism is nullptr.");
    }

    this->A = A.has_value() ? A.value() : this->proposalImpl->getA();
    this->b = b.has_value() ? b.value() : this->proposalImpl->getB();

    VectorType parameterState = this->proposalImpl->getState();
    this->activationState = Eigen::VectorXd::Ones(parameterState.rows());
    // precomputes backward & forwards distances. Works because we use uniform jumping distribution.
    this->backwardDistances = VectorType::Zero(this->jumpIndices.rows());
    this->forwardDistances = VectorType::Zero(this->jumpIndices.rows());
    for (long i = 0; i < this->jumpIndices.rows(); ++i) {
        parameterState(this->jumpIndices(i)) = this->defaultValues(i);
        // Starts with all optional parameters deactivated, which is the simplest model
        this->activationState(jumpIndices(i)) = 0.;
        auto[b, f] = distanceInCoordinateDirection(this->A,
                                                   this->b,
                                                   this->defaultValues(i),
                                                   this->jumpIndices(i));
        this->backwardDistances(i) = b;
        this->forwardDistances(i) = f;
    }
    this->activationProposal = activationState;

    this->proposalImpl->setState(parameterState);
    this->logAcceptanceChanceModelJump = 0;
    this->proposal = VectorType::Zero(activationProposal.rows() + parameterState.rows());
    this->lastProposalJumpedModel = false;
}

hops::ReversibleJumpProposal::ReversibleJumpProposal(const ReversibleJumpProposal &other) {
    this->proposalImpl = other.proposalImpl->copyProposal();
    this->jumpIndices = other.jumpIndices;
    this->defaultValues = other.defaultValues;
    this->activationState = other.activationState;
    this->activationProposal = other.activationProposal;
    this->proposal = other.proposal;
    this->lastProposalJumpedModel = other.lastProposalJumpedModel;
    this->logAcceptanceChanceModelJump = other.logAcceptanceChanceModelJump;
    this->backwardDistances = other.backwardDistances;
    this->forwardDistances = other.forwardDistances;
    this->A = other.A;
    this->b = other.b;
}

hops::ReversibleJumpProposal::ReversibleJumpProposal(ReversibleJumpProposal &&other) noexcept {
    this->proposalImpl = std::move(other.proposalImpl);
    this->jumpIndices = other.jumpIndices;
    this->defaultValues = other.defaultValues;
    this->activationState = other.activationState;
    this->activationProposal = other.activationProposal;
    this->proposal = other.proposal;
    this->lastProposalJumpedModel = other.lastProposalJumpedModel;
    this->logAcceptanceChanceModelJump = other.logAcceptanceChanceModelJump;
    this->backwardDistances = other.backwardDistances;
    this->forwardDistances = other.forwardDistances;
    this->A = other.A;
    this->b = other.b;
}

hops::ReversibleJumpProposal &hops::ReversibleJumpProposal::operator=(const ReversibleJumpProposal &other) {
    this->proposalImpl = other.proposalImpl->copyProposal();
    this->jumpIndices = other.jumpIndices;
    this->defaultValues = other.defaultValues;
    this->activationState = other.activationState;
    this->activationProposal = other.activationProposal;
    this->proposal = other.proposal;
    this->lastProposalJumpedModel = other.lastProposalJumpedModel;
    this->logAcceptanceChanceModelJump = other.logAcceptanceChanceModelJump;
    this->backwardDistances = other.backwardDistances;
    this->forwardDistances = other.forwardDistances;
    this->A = other.A;
    this->b = other.b;
    return *this;
}

hops::ReversibleJumpProposal &hops::ReversibleJumpProposal::operator=(ReversibleJumpProposal &&other) noexcept {
    this->proposalImpl = std::move(other.proposalImpl);
    this->jumpIndices = other.jumpIndices;
    this->defaultValues = other.defaultValues;
    this->activationState = other.activationState;
    this->activationProposal = other.activationProposal;
    this->proposal = other.proposal;
    this->lastProposalJumpedModel = other.lastProposalJumpedModel;
    this->logAcceptanceChanceModelJump = other.logAcceptanceChanceModelJump;
    this->backwardDistances = other.backwardDistances;
    this->forwardDistances = other.forwardDistances;
    this->A = other.A;
    this->b = other.b;
    return *this;
}

hops::VectorType &hops::ReversibleJumpProposal::propose(RandomNumberGenerator &rng) {
    if (uniformRealDistribution(rng) < modelJumpProbability) {
        lastProposalJumpedModel = true;
        proposal = proposeModel(rng);
    } else {
        lastProposalJumpedModel = false;
        this->activationProposal = this->activationState;
        proposal = wrapProposal(proposalImpl->propose(rng, activationState));
    }
    return proposal;
}

double hops::ReversibleJumpProposal::computeLogAcceptanceProbability() {
    if (lastProposalJumpedModel) {
        return this->logAcceptanceChanceModelJump;
    }
    return proposalImpl->computeLogAcceptanceProbability();
}

hops::VectorType &hops::ReversibleJumpProposal::acceptProposal() {
    activationState = activationProposal;
    return wrapProposal(proposalImpl->acceptProposal());
}

void hops::ReversibleJumpProposal::setState(const VectorType &state) {
    this->activationState = state.topRows(activationState.rows());
    proposalImpl->setState(state.bottomRows(state.rows() - this->activationState.rows()));
}

void hops::ReversibleJumpProposal::setProposal(const VectorType &newProposal) {
    proposal = newProposal;
    activationProposal = newProposal.topRows(this->activationProposal.rows());
    proposalImpl->setProposal(proposal.bottomRows(proposal.rows() - this->activationProposal.rows()));
}

hops::VectorType hops::ReversibleJumpProposal::getState() const {
    VectorType parameterState = proposalImpl->getState();
    VectorType state(activationState.rows() + parameterState.rows());
    state << activationState, parameterState;
    return state;
}

hops::VectorType hops::ReversibleJumpProposal::getProposal() const {
    return proposal;
}

std::vector<std::string> hops::ReversibleJumpProposal::getParameterNames() const {
    std::vector<std::string> parameterNames = proposalImpl->getParameterNames();
    parameterNames.emplace_back(
            ProposalParameterName[static_cast<int>(ProposalParameter::MODEL_JUMP_PROBABILITY)]);
    parameterNames.emplace_back(
            ProposalParameterName[static_cast<int>(ProposalParameter::ACTIVATION_PROBABILITY)]);
    parameterNames.emplace_back(
            ProposalParameterName[static_cast<int>(ProposalParameter::DEACTIVATION_PROBABILITY)]);
    return parameterNames;
}

std::any hops::ReversibleJumpProposal::getParameter(const ProposalParameter &parameter) const {
    switch (parameter) {
        case ProposalParameter::MODEL_JUMP_PROBABILITY:
            return modelJumpProbability;
        case ProposalParameter::ACTIVATION_PROBABILITY:
            return activationProbability;
        case ProposalParameter::DEACTIVATION_PROBABILITY:
            return deactivationProbability;
        default:
            return proposalImpl->getParameter(parameter);
    }
}

std::string hops::ReversibleJumpProposal::getParameterType(const ProposalParameter &parameter) const {
    if (parameter == ProposalParameter::MODEL_JUMP_PROBABILITY ||
        parameter == ProposalParameter::ACTIVATION_PROBABILITY ||
        parameter == ProposalParameter::DEACTIVATION_PROBABILITY) {
        return "double";
    } else {
        return proposalImpl->getParameterType(parameter);
    }
}

void hops::ReversibleJumpProposal::setParameter(const ProposalParameter &parameter, const std::any &value) {
    switch (parameter) {
        case ProposalParameter::MODEL_JUMP_PROBABILITY:
            if (std::any_cast<double>(value) >= 1.) {
                throw std::invalid_argument(std::string(
                        ProposalParameterName[static_cast<int>(ProposalParameter::MODEL_JUMP_PROBABILITY)]) +
                                            " can not be set to 1 or larger.");
            }
            modelJumpProbability = std::any_cast<double>(value);
            break;
        case ProposalParameter::ACTIVATION_PROBABILITY:
            if (std::any_cast<double>(value) >= 1.) {
                throw std::invalid_argument(std::string(
                        ProposalParameterName[static_cast<int>(ProposalParameter::ACTIVATION_PROBABILITY)]) +
                                            " can not be set to 1 or larger.");
            }
            activationProbability = std::any_cast<double>(value);
            break;
        case ProposalParameter::DEACTIVATION_PROBABILITY:
            if (std::any_cast<double>(value) >= 1.) {
                throw std::invalid_argument(std::string(
                        ProposalParameterName[static_cast<int>(ProposalParameter::DEACTIVATION_PROBABILITY)]) +
                                            " can not be set to 1 or larger.");
            }
            deactivationProbability = std::any_cast<double>(value);
            break;
        default:
            proposalImpl->setParameter(parameter, value);
    }
}

std::string hops::ReversibleJumpProposal::getProposalName() const {
    return "RJMCMC(" + proposalImpl->getProposalName() + ")";
}

std::unique_ptr<hops::Proposal> hops::ReversibleJumpProposal::copyProposal() const {
    return std::make_unique<hops::ReversibleJumpProposal>(*this);
}

const hops::MatrixType &hops::ReversibleJumpProposal::getA() const {
    return this->A;
}

const hops::VectorType &hops::ReversibleJumpProposal::getB() const {
    return this->b;
}

hops::VectorType &hops::ReversibleJumpProposal::proposeModel(RandomNumberGenerator &randomNumberGenerator) {
    VectorType parameterProposal = proposalImpl->getState();
    this->activationProposal = this->activationState;
    this->logAcceptanceChanceModelJump = 0;

    for (long i = 0; i < jumpIndices.rows(); ++i) {
        long jumpIndex = this->jumpIndices(i);
        bool isActive = this->activationState(jumpIndex) != 0;
        auto jumpProbability = isActive ? this->deactivationProbability
                                        : this->activationProbability;

        if (uniformRealDistribution(randomNumberGenerator) < jumpProbability) {
            this->activationProposal(jumpIndex) = !isActive;
        } else {
            this->activationProposal(jumpIndex) = isActive;
        }

        if (!this->activationProposal(jumpIndex) && this->activationState(jumpIndex)) {
            // deactivate
            logAcceptanceChanceModelJump += std::log(activationProbability);
            logAcceptanceChanceModelJump -= std::log(deactivationProbability);
            parameterProposal(jumpIndex) = defaultValues(i);
        } else if (this->activationProposal(jumpIndex) && !this->activationState(jumpIndex)) {
            // activate
            logAcceptanceChanceModelJump -= std::log(activationProbability);
            logAcceptanceChanceModelJump += std::log(deactivationProbability);
            parameterProposal(jumpIndex) = defaultValues(i) + stepDistribution.draw(randomNumberGenerator,
                                                                                    backwardDistances(i),
                                                                                    forwardDistances(i));
        }
    }

    proposalImpl->setProposal(parameterProposal);
    this->logAcceptanceChanceModelJump += proposalImpl->getStateNegativeLogLikelihood()
                                          - proposalImpl->getProposalNegativeLogLikelihood();

    return wrapProposal(parameterProposal);
}

std::optional<double> hops::ReversibleJumpProposal::getStepSize() const {
    return this->proposalImpl->getStepSize();
}

hops::VectorType &
hops::ReversibleJumpProposal::propose(RandomNumberGenerator &rng, const Eigen::VectorXd &activeIndices) {
    return wrapProposal(proposalImpl->propose(rng, activeIndices));
}

hops::VectorType &hops::ReversibleJumpProposal::wrapProposal(const VectorType &parameterProposal) {
    proposal.setZero();
    proposal << this->activationProposal, parameterProposal;
    return proposal;
}

void hops::ReversibleJumpProposal::setDimensionNames(const std::vector<std::string> &names) {
    proposalImpl->setDimensionNames(names);
}

std::vector<std::string> hops::ReversibleJumpProposal::getDimensionNames() const {
    // Vector is constructed on demand, because it typically is not used repeatedly.
    std::vector<std::string> dimensionNames = proposalImpl->getDimensionNames();
    std::vector<std::string> names;
    for (long i = 0; i < this->activationState.rows(); ++i) {
        names.emplace_back(dimensionNames[i] + "_activation");
    }
    names.insert(names.end(), dimensionNames.begin(), dimensionNames.end());
    return names;
}

const std::unique_ptr<hops::Proposal> &hops::ReversibleJumpProposal::getProposalImpl() const {
    return proposalImpl;
}
void hops::ReversibleJumpProposal::setProposalImpl(std::unique_ptr<Proposal> proposalImpl) {
    ReversibleJumpProposal::proposalImpl = std::move(proposalImpl);
}
double hops::ReversibleJumpProposal::getModelJumpProbability() const {
    return modelJumpProbability;
}
void hops::ReversibleJumpProposal::setModelJumpProbability(double modelJumpProbability) {
    ReversibleJumpProposal::modelJumpProbability = modelJumpProbability;
}
double hops::ReversibleJumpProposal::getActivationProbability() const {
    return activationProbability;
}
void hops::ReversibleJumpProposal::setActivationProbability(double activationProbability) {
    ReversibleJumpProposal::activationProbability = activationProbability;
}
double hops::ReversibleJumpProposal::getDeactivationProbability() const {
    return deactivationProbability;
}
void hops::ReversibleJumpProposal::setDeactivationProbability(double deactivationProbability) {
    ReversibleJumpProposal::deactivationProbability = deactivationProbability;
}
const hops::VectorType &hops::ReversibleJumpProposal::getBackwardDistances() const {
    return backwardDistances;
}
void hops::ReversibleJumpProposal::setBackwardDistances(const hops::VectorType &backwardDistances) {
    ReversibleJumpProposal::backwardDistances = backwardDistances;
}
const hops::VectorType &hops::ReversibleJumpProposal::getForwardDistances() const {
    return forwardDistances;
}
void hops::ReversibleJumpProposal::setForwardDistances(const hops::VectorType &forwardDistances) {
    ReversibleJumpProposal::forwardDistances = forwardDistances;
}
const Eigen::VectorXi &hops::ReversibleJumpProposal::getJumpIndices() const {
    return jumpIndices;
}
void hops::ReversibleJumpProposal::setJumpIndices(const Eigen::VectorXi &jumpIndices) {
    ReversibleJumpProposal::jumpIndices = jumpIndices;
}
const hops::VectorType &hops::ReversibleJumpProposal::getDefaultValues() const {
    return defaultValues;
}
void hops::ReversibleJumpProposal::setDefaultValues(const hops::VectorType &defaultValues) {
    ReversibleJumpProposal::defaultValues = defaultValues;
}
const hops::VectorType &hops::ReversibleJumpProposal::getActivationState() const {
    return activationState;
}
void hops::ReversibleJumpProposal::setActivationState(const hops::VectorType &activationState) {
    ReversibleJumpProposal::activationState = activationState;
}
const hops::VectorType &hops::ReversibleJumpProposal::getActivationProposal() const {
    return activationProposal;
}
void hops::ReversibleJumpProposal::setActivationProposal(const hops::VectorType &activationProposal) {
    ReversibleJumpProposal::activationProposal = activationProposal;
}
double hops::ReversibleJumpProposal::getLogAcceptanceChanceModelJump() const {
    return logAcceptanceChanceModelJump;
}
void hops::ReversibleJumpProposal::setLogAcceptanceChanceModelJump(double logAcceptanceChanceModelJump) {
    ReversibleJumpProposal::logAcceptanceChanceModelJump = logAcceptanceChanceModelJump;
}
bool hops::ReversibleJumpProposal::isLastProposalJumpedModel() const {
    return lastProposalJumpedModel;
}
void hops::ReversibleJumpProposal::setLastProposalJumpedModel(bool lastProposalJumpedModel) {
    ReversibleJumpProposal::lastProposalJumpedModel = lastProposalJumpedModel;
}

