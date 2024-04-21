#include "AcceptanceRateTarget.hpp"

#include <utility>

hops::AcceptanceRateTarget::AcceptanceRateTarget(std::vector<std::shared_ptr<MarkovChain>> markovChains,
                                                 unsigned long numberOfTestSamples, double acceptanceRateTargetValue,
                                                 unsigned long order) : m_markovChains(std::move(markovChains)),
                                                                        m_numberOfTestSamples(numberOfTestSamples),
                                                                        m_acceptanceRateTargetValue(acceptanceRateTargetValue),
        m_order(order) {}

std::string hops::AcceptanceRateTarget::getName() const {
    return "AcceptanceRate";
}

std::unique_ptr<hops::TuningTarget> hops::AcceptanceRateTarget::copyTuningTarget() const {
    return std::make_unique<AcceptanceRateTarget>(*this);
}

std::pair<double, double> hops::AcceptanceRateTarget::operator()(const hops::VectorType &x,
                                                                 const std::vector<RandomNumberGenerator *> &randomNumberGenerators) {
    if (m_markovChains.size() != randomNumberGenerators.size()) {
        throw std::runtime_error("Number of random number generators must match number of markov chains.");
    }

    double stepSize = std::pow(10, x(0));
    std::vector<double> acceptanceRateScores(m_markovChains.size());
    for (size_t i = 0; i < m_markovChains.size(); ++i) {
        m_markovChains[i]->setParameter(ProposalParameter::STEP_SIZE, stepSize);

        double acceptanceRate = std::get<0>(m_markovChains[i]->draw(*randomNumberGenerators[i], m_numberOfTestSamples));

        double deltaScale = (
                acceptanceRate > m_acceptanceRateTargetValue ?
                1 - m_acceptanceRateTargetValue : m_acceptanceRateTargetValue);
        acceptanceRateScores[i] =
                1 - std::pow(std::abs(acceptanceRate - m_acceptanceRateTargetValue), m_order) / std::pow(deltaScale, m_order);
    }

    double mean = std::accumulate(acceptanceRateScores.begin(), acceptanceRateScores.end(), 0.0) /
                  acceptanceRateScores.size();

    double squaredSum = std::inner_product(acceptanceRateScores.begin(), acceptanceRateScores.end(),
                                           acceptanceRateScores.begin(), 0.0);
    double error = squaredSum / acceptanceRateScores.size() - mean * mean;

    return {mean, error};
}

