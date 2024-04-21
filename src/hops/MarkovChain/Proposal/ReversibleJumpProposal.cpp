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
                                                     const std::optional<Eigen::VectorXd> &b) : m_proposalImpl(std::move(proposalImpl)),
                                                                                                m_jumpIndices(jumpIndices),
                                                                                                m_defaultValues(parameterDefaultValues) {

    if (this->m_jumpIndices.rows() != this->m_defaultValues.rows()) {
        throw std::runtime_error("dimension missmatch in input");
    }
    if (!this->m_proposalImpl) {
        throw std::runtime_error("m_proposal mechanism is nullptr.");
    }

    this->m_A = A.has_value() ? A.value() : this->m_proposalImpl->getA();
    this->m_b = b.has_value() ? b.value() : this->m_proposalImpl->getB();

    VectorType parameterState = this->m_proposalImpl->getState();
    this->m_activationState = Eigen::VectorXd::Ones(parameterState.rows());
    // precomputes backward & forwards distances. Works because we use uniform jumping distribution.
    this->m_backwardDistances = VectorType::Zero(this->m_jumpIndices.rows());
    this->m_forwarDistances = VectorType::Zero(this->m_jumpIndices.rows());
    for (long i = 0; i < this->m_jumpIndices.rows(); ++i) {
        parameterState(this->m_jumpIndices(i)) = this->m_defaultValues(i);
        // Starts with all optional parameters deactivated, which is the simplest model
        this->m_activationState(jumpIndices(i)) = 0.;
        auto[backwards, forwards] = distanceInCoordinateDirection(this->m_A,
                                                   this->m_b,
                                                   this->m_defaultValues(i),
                                                   this->m_jumpIndices(i));
        this->m_backwardDistances(i) = backwards;
        this->m_forwarDistances(i) = forwards;
    }
    this->m_activationProposal = m_activationState;

    this->m_proposalImpl->setState(parameterState);
    this->m_logAcceptanceChanceModelJump = 0;
    this->m_proposal = VectorType::Zero(m_activationProposal.rows() + parameterState.rows());
    this->m_lastProposalJumpedModel = false;
}

hops::ReversibleJumpProposal::ReversibleJumpProposal(const ReversibleJumpProposal &other) {
    this->m_proposalImpl = other.m_proposalImpl->copyProposal();
    this->m_jumpIndices = other.m_jumpIndices;
    this->m_defaultValues = other.m_defaultValues;
    this->m_activationState = other.m_activationState;
    this->m_activationProposal = other.m_activationProposal;
    this->m_proposal = other.m_proposal;
    this->m_lastProposalJumpedModel = other.m_lastProposalJumpedModel;
    this->m_logAcceptanceChanceModelJump = other.m_logAcceptanceChanceModelJump;
    this->m_backwardDistances = other.m_backwardDistances;
    this->m_forwarDistances = other.m_forwarDistances;
    this->m_A = other.m_A;
    this->m_b = other.m_b;
}

hops::ReversibleJumpProposal::ReversibleJumpProposal(ReversibleJumpProposal &&other) noexcept {
    this->m_proposalImpl = std::move(other.m_proposalImpl);
    this->m_jumpIndices = other.m_jumpIndices;
    this->m_defaultValues = other.m_defaultValues;
    this->m_activationState = other.m_activationState;
    this->m_activationProposal = other.m_activationProposal;
    this->m_proposal = other.m_proposal;
    this->m_lastProposalJumpedModel = other.m_lastProposalJumpedModel;
    this->m_logAcceptanceChanceModelJump = other.m_logAcceptanceChanceModelJump;
    this->m_backwardDistances = other.m_backwardDistances;
    this->m_forwarDistances = other.m_forwarDistances;
    this->m_A = other.m_A;
    this->m_b = other.m_b;
}

hops::ReversibleJumpProposal &hops::ReversibleJumpProposal::operator=(const ReversibleJumpProposal &other) {
    this->m_proposalImpl = other.m_proposalImpl->copyProposal();
    this->m_jumpIndices = other.m_jumpIndices;
    this->m_defaultValues = other.m_defaultValues;
    this->m_activationState = other.m_activationState;
    this->m_activationProposal = other.m_activationProposal;
    this->m_proposal = other.m_proposal;
    this->m_lastProposalJumpedModel = other.m_lastProposalJumpedModel;
    this->m_logAcceptanceChanceModelJump = other.m_logAcceptanceChanceModelJump;
    this->m_backwardDistances = other.m_backwardDistances;
    this->m_forwarDistances = other.m_forwarDistances;
    this->m_A = other.m_A;
    this->m_b = other.m_b;
    return *this;
}

hops::ReversibleJumpProposal &hops::ReversibleJumpProposal::operator=(ReversibleJumpProposal &&other) noexcept {
    this->m_proposalImpl = std::move(other.m_proposalImpl);
    this->m_jumpIndices = other.m_jumpIndices;
    this->m_defaultValues = other.m_defaultValues;
    this->m_activationState = other.m_activationState;
    this->m_activationProposal = other.m_activationProposal;
    this->m_proposal = other.m_proposal;
    this->m_lastProposalJumpedModel = other.m_lastProposalJumpedModel;
    this->m_logAcceptanceChanceModelJump = other.m_logAcceptanceChanceModelJump;
    this->m_backwardDistances = other.m_backwardDistances;
    this->m_forwarDistances = other.m_forwarDistances;
    this->m_A = other.m_A;
    this->m_b = other.m_b;
    return *this;
}

hops::VectorType &hops::ReversibleJumpProposal::propose(RandomNumberGenerator &rng) {
    if (uniformRealDistribution(rng) < m_modelJumpProbability) {
        m_lastProposalJumpedModel = true;
        m_proposal = proposeModel(rng);
    } else {
        m_lastProposalJumpedModel = false;
        this->m_activationProposal = this->m_activationState;
        m_proposal = wrapProposal(m_proposalImpl->propose(rng, m_activationState));
    }
    return m_proposal;
}

double hops::ReversibleJumpProposal::computeLogAcceptanceProbability() {
    if (m_lastProposalJumpedModel) {
        return this->m_logAcceptanceChanceModelJump;
    }
    return m_proposalImpl->computeLogAcceptanceProbability();
}

hops::VectorType &hops::ReversibleJumpProposal::acceptProposal() {
    m_activationState = m_activationProposal;
    return wrapProposal(m_proposalImpl->acceptProposal());
}

void hops::ReversibleJumpProposal::setState(const VectorType &state) {
    this->m_activationState = state.topRows(m_activationState.rows());
    m_proposalImpl->setState(state.bottomRows(state.rows() - this->m_activationState.rows()));
}

void hops::ReversibleJumpProposal::setProposal(const VectorType &newProposal) {
    m_proposal = newProposal;
    m_activationProposal = newProposal.topRows(this->m_activationProposal.rows());
    m_proposalImpl->setProposal(m_proposal.bottomRows(m_proposal.rows() - this->m_activationProposal.rows()));
}

hops::VectorType hops::ReversibleJumpProposal::getState() const {
    VectorType parameterState = m_proposalImpl->getState();
    VectorType state(m_activationState.rows() + parameterState.rows());
    state << m_activationState, parameterState;
    return state;
}

hops::VectorType hops::ReversibleJumpProposal::getProposal() const {
    return m_proposal;
}

std::vector<std::string> hops::ReversibleJumpProposal::getParameterNames() const {
    std::vector<std::string> parameterNames = m_proposalImpl->getParameterNames();
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
            return m_modelJumpProbability;
        case ProposalParameter::ACTIVATION_PROBABILITY:
            return m_activiationProbability;
        case ProposalParameter::DEACTIVATION_PROBABILITY:
            return m_deativationProbability;
        default:
            return m_proposalImpl->getParameter(parameter);
    }
}

std::string hops::ReversibleJumpProposal::getParameterType(const ProposalParameter &parameter) const {
    if (parameter == ProposalParameter::MODEL_JUMP_PROBABILITY ||
        parameter == ProposalParameter::ACTIVATION_PROBABILITY ||
        parameter == ProposalParameter::DEACTIVATION_PROBABILITY) {
        return "double";
    } else {
        return m_proposalImpl->getParameterType(parameter);
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
            m_modelJumpProbability = std::any_cast<double>(value);
            break;
        case ProposalParameter::ACTIVATION_PROBABILITY:
            if (std::any_cast<double>(value) >= 1.) {
                throw std::invalid_argument(std::string(
                        ProposalParameterName[static_cast<int>(ProposalParameter::ACTIVATION_PROBABILITY)]) +
                                            " can not be set to 1 or larger.");
            }
            m_activiationProbability = std::any_cast<double>(value);
            break;
        case ProposalParameter::DEACTIVATION_PROBABILITY:
            if (std::any_cast<double>(value) >= 1.) {
                throw std::invalid_argument(std::string(
                        ProposalParameterName[static_cast<int>(ProposalParameter::DEACTIVATION_PROBABILITY)]) +
                                            " can not be set to 1 or larger.");
            }
            m_deativationProbability = std::any_cast<double>(value);
            break;
        default:
            m_proposalImpl->setParameter(parameter, value);
    }
}

std::string hops::ReversibleJumpProposal::getProposalName() const {
    return "RJMCMC(" + m_proposalImpl->getProposalName() + ")";
}

std::unique_ptr<hops::Proposal> hops::ReversibleJumpProposal::copyProposal() const {
    return std::make_unique<hops::ReversibleJumpProposal>(*this);
}

const hops::MatrixType &hops::ReversibleJumpProposal::getA() const {
    return this->m_A;
}

const hops::VectorType &hops::ReversibleJumpProposal::getB() const {
    return this->m_b;
}

hops::VectorType &hops::ReversibleJumpProposal::proposeModel(RandomNumberGenerator &randomNumberGenerator) {
    VectorType parameterProposal = m_proposalImpl->getState();
    this->m_activationProposal = this->m_activationState;
    this->m_logAcceptanceChanceModelJump = 0;

    for (long i = 0; i < m_jumpIndices.rows(); ++i) {
        long jumpIndex = this->m_jumpIndices(i);
        bool isActive = this->m_activationState(jumpIndex) != 0;
        auto jumpProbability = isActive ? this->m_deativationProbability
                                        : this->m_activiationProbability;

        if (uniformRealDistribution(randomNumberGenerator) < jumpProbability) {
            this->m_activationProposal(jumpIndex) = !isActive;
        } else {
            this->m_activationProposal(jumpIndex) = isActive;
        }

        if (!this->m_activationProposal(jumpIndex) && this->m_activationState(jumpIndex)) {
            // deactivate
            m_logAcceptanceChanceModelJump += std::log(m_activiationProbability);
            m_logAcceptanceChanceModelJump -= std::log(m_deativationProbability);
            parameterProposal(jumpIndex) = m_defaultValues(i);
        } else if (this->m_activationProposal(jumpIndex) && !this->m_activationState(jumpIndex)) {
            // activate
            m_logAcceptanceChanceModelJump -= std::log(m_activiationProbability);
            m_logAcceptanceChanceModelJump += std::log(m_deativationProbability);
            parameterProposal(jumpIndex) = m_defaultValues(i) + stepDistribution.draw(randomNumberGenerator,
                                                                                    m_backwardDistances(i),
                                                                                    m_forwarDistances(i));
        }
    }

    m_proposalImpl->setProposal(parameterProposal);
    this->m_logAcceptanceChanceModelJump += m_proposalImpl->getStateNegativeLogLikelihood()
                                          - m_proposalImpl->getProposalNegativeLogLikelihood();

    return wrapProposal(parameterProposal);
}

std::optional<double> hops::ReversibleJumpProposal::getStepSize() const {
    return this->m_proposalImpl->getStepSize();
}

hops::VectorType &
hops::ReversibleJumpProposal::propose(RandomNumberGenerator &rng, const Eigen::VectorXd &activeIndices) {
    return wrapProposal(m_proposalImpl->propose(rng, activeIndices));
}

hops::VectorType &hops::ReversibleJumpProposal::wrapProposal(const VectorType &parameterProposal) {
    m_proposal.setZero();
    m_proposal << this->m_activationProposal, parameterProposal;
    return m_proposal;
}

void hops::ReversibleJumpProposal::setDimensionNames(const std::vector<std::string> &names) {
    m_proposalImpl->setDimensionNames(names);
}

std::vector<std::string> hops::ReversibleJumpProposal::getDimensionNames() const {
    // Vector is constructed on demand, because it typically is not used repeatedly.
    std::vector<std::string> dimensionNames = m_proposalImpl->getDimensionNames();
    std::vector<std::string> names;
    for (long i = 0; i < this->m_activationState.rows(); ++i) {
        names.emplace_back(dimensionNames[i] + "_activation");
    }
    names.insert(names.end(), dimensionNames.begin(), dimensionNames.end());
    return names;
}

const std::unique_ptr<hops::Proposal> &hops::ReversibleJumpProposal::getProposalImpl() const {
    return m_proposalImpl;
}
void hops::ReversibleJumpProposal::setProposalImpl(std::unique_ptr<Proposal> proposalImpl) {
    ReversibleJumpProposal::m_proposalImpl = std::move(proposalImpl);
}
double hops::ReversibleJumpProposal::getModelJumpProbability() const {
    return m_modelJumpProbability;
}
void hops::ReversibleJumpProposal::setModelJumpProbability(double modelJumpProbability) {
    ReversibleJumpProposal::m_modelJumpProbability = modelJumpProbability;
}
double hops::ReversibleJumpProposal::getActivationProbability() const {
    return m_activiationProbability;
}
void hops::ReversibleJumpProposal::setActivationProbability(double activationProbability) {
    ReversibleJumpProposal::m_activiationProbability = activationProbability;
}
double hops::ReversibleJumpProposal::getDeactivationProbability() const {
    return m_deativationProbability;
}
void hops::ReversibleJumpProposal::setDeactivationProbability(double deactivationProbability) {
    ReversibleJumpProposal::m_deativationProbability = deactivationProbability;
}
const hops::VectorType &hops::ReversibleJumpProposal::getBackwardDistances() const {
    return m_backwardDistances;
}
void hops::ReversibleJumpProposal::setBackwardDistances(const hops::VectorType &backwardDistances) {
    ReversibleJumpProposal::m_backwardDistances = backwardDistances;
}
const hops::VectorType &hops::ReversibleJumpProposal::getForwardDistances() const {
    return m_forwarDistances;
}
void hops::ReversibleJumpProposal::setForwardDistances(const hops::VectorType &forwardDistances) {
    ReversibleJumpProposal::m_forwarDistances = forwardDistances;
}
const Eigen::VectorXi &hops::ReversibleJumpProposal::getJumpIndices() const {
    return m_jumpIndices;
}
void hops::ReversibleJumpProposal::setJumpIndices(const Eigen::VectorXi &jumpIndices) {
    ReversibleJumpProposal::m_jumpIndices = jumpIndices;
}
const hops::VectorType &hops::ReversibleJumpProposal::getDefaultValues() const {
    return m_defaultValues;
}
void hops::ReversibleJumpProposal::setDefaultValues(const hops::VectorType &defaultValues) {
    ReversibleJumpProposal::m_defaultValues = defaultValues;
}
const hops::VectorType &hops::ReversibleJumpProposal::getActivationState() const {
    return m_activationState;
}
void hops::ReversibleJumpProposal::setActivationState(const hops::VectorType &activationState) {
    ReversibleJumpProposal::m_activationState = activationState;
}
const hops::VectorType &hops::ReversibleJumpProposal::getActivationProposal() const {
    return m_activationProposal;
}
void hops::ReversibleJumpProposal::setActivationProposal(const hops::VectorType &activationProposal) {
    ReversibleJumpProposal::m_activationProposal = activationProposal;
}
double hops::ReversibleJumpProposal::getLogAcceptanceChanceModelJump() const {
    return m_logAcceptanceChanceModelJump;
}
void hops::ReversibleJumpProposal::setLogAcceptanceChanceModelJump(double logAcceptanceChanceModelJump) {
    ReversibleJumpProposal::m_logAcceptanceChanceModelJump = logAcceptanceChanceModelJump;
}
bool hops::ReversibleJumpProposal::isLastProposalJumpedModel() const {
    return m_lastProposalJumpedModel;
}
void hops::ReversibleJumpProposal::setLastProposalJumpedModel(bool lastProposalJumpedModel) {
    ReversibleJumpProposal::m_lastProposalJumpedModel = lastProposalJumpedModel;
}

