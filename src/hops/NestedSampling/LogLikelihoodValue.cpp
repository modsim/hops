#include "LogLikelihoodValue.hpp"

double hops::LogLikelihoodValue::getValue() const {
    return value;
}

double hops::LogLikelihoodValue::getTiebreaker() const {
    return tiebreaker;
}

bool hops::LogLikelihoodValue::operator<(const hops::LogLikelihoodValue &other) const {
    if(value == other.value) {
        return tiebreaker < other.tiebreaker;
    }
    return value < other.value;
}

hops::LogLikelihoodValue::LogLikelihoodValue(double value, double tiebreaker) : value(value), tiebreaker(tiebreaker) {}

void hops::LogLikelihoodValue::setTiebreaker(double newTiebreaker) {
    LogLikelihoodValue::tiebreaker = newTiebreaker;
}
