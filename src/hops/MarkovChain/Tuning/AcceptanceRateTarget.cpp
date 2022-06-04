#include "AcceptanceRateTarget.hpp"

#include <utility>

hops::AcceptanceRateTarget::AcceptanceRateTarget(std::vector<std::shared_ptr<MarkovChain>> markovChains,
                                                 unsigned long numberOfTestSamples, double acceptanceRateTargetValue,
                                                 unsigned long order) :
        markovChains(std::move(markovChains)),
        numberOfTestSamples(numberOfTestSamples),
        acceptanceRateTargetValue(acceptanceRateTargetValue),
        order(order) {}

std::string hops::AcceptanceRateTarget::getName() const {
    return "AcceptanceRate";
}

std::unique_ptr<hops::TuningTarget> hops::AcceptanceRateTarget::copyTuningTarget() const {
    return std::make_unique<AcceptanceRateTarget>(*this);
}

std::pair<double, double> hops::AcceptanceRateTarget::operator()(const hops::VectorType &x,
                                                                 const std::vector<RandomNumberGenerator *> &randomNumberGenerators) {
    if (markovChains.size() != randomNumberGenerators.size()) {
        throw std::runtime_error("Number of random number generators must match number of markov chains.");
    }

    double stepSize = std::pow(10, x(0));
    std::vector<double> acceptanceRateScores(markovChains.size());
    for (size_t i = 0; i < markovChains.size(); ++i) {
        markovChains[i]->setParameter(ProposalParameter::STEP_SIZE, stepSize);

        double acceptanceRate = std::get<0>(markovChains[i]->draw(*randomNumberGenerators[i], numberOfTestSamples));

        double deltaScale = (
                acceptanceRate > acceptanceRateTargetValue ?
                1 - acceptanceRateTargetValue :
                acceptanceRateTargetValue
        );
        acceptanceRateScores[i] =
                1 - std::pow(std::abs(acceptanceRate - acceptanceRateTargetValue), order) / std::pow(deltaScale, order);
    }

    double mean = std::accumulate(acceptanceRateScores.begin(), acceptanceRateScores.end(), 0.0) /
                  acceptanceRateScores.size();

    double squaredSum = std::inner_product(acceptanceRateScores.begin(), acceptanceRateScores.end(),
                                           acceptanceRateScores.begin(), 0.0);
    double error = squaredSum / acceptanceRateScores.size() - mean * mean;

    return {mean, error};
}

