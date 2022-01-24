#include "ProposalStatistics.hpp"

void hops::ProposalStatistics::appendInfo(const std::string &name, double value) {
    infos[name].emplace_back(value);
}
const std::unordered_map<std::string, std::vector<double>> &hops::ProposalStatistics::getStatistics() const {
    return infos;
}


