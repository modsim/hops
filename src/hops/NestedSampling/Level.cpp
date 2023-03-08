#include "Level.hpp"

std::vector<std::string> hops::Level::getDimensionNames() {
    return {"log_X", "log_likelihood", "tiebreaker", "accepts", "tries", "exceeds", "visits"};
}

hops::VectorType hops::Level::asVector() {
    VectorType vector(7);
    vector << log_X, likelihood.getValue(), likelihood.getTiebreaker(), accepts, tries, exceeds, visits;
    return vector;
}

hops::Level::Level(const hops::LogLikelihoodValue &likelihood) :
        log_X(0.),
        likelihood(likelihood),
        accepts(0),
        tries(0),
        exceeds(0),
        visits(0) {}

double hops::Level::getLogX() const {
    return log_X;
}

void hops::Level::setLogX(double logX) {
    log_X = logX;
}

const hops::LogLikelihoodValue &hops::Level::getLikelihood() const {
    return likelihood;
}

void hops::Level::setLikelihood(const hops::LogLikelihoodValue &likelihood) {
    Level::likelihood = likelihood;
}

unsigned long long int hops::Level::getAccepts() const {
    return accepts;
}

void hops::Level::setAccepts(unsigned long long int accepts) {
    Level::accepts = accepts;
}

unsigned long long int hops::Level::getTries() const {
    return tries;
}

void hops::Level::setTries(unsigned long long int tries) {
    Level::tries = tries;
}

unsigned long long int hops::Level::getExceeds() const {
    return exceeds;
}

void hops::Level::setExceeds(unsigned long long int exceeds) {
    Level::exceeds = exceeds;
}

unsigned long long int hops::Level::getVisits() const {
    return visits;
}

void hops::Level::setVisits(unsigned long long int visits) {
    Level::visits = visits;
}

void hops::Level::incrementAccepts(unsigned long long int increment) {
    accepts += increment;
}

void hops::Level::incrementTries(unsigned long long int increment) {
    tries += increment;
}

void hops::Level::incrementExceeds(unsigned long long int increment) {
    exceeds += increment;
}

void hops::Level::incrementVisits(unsigned long long int increment) {
    visits += increment;
}

