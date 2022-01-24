#ifndef HOPS_PROPOSALSTATISTICS_HPP
#define HOPS_PROPOSALSTATISTICS_HPP

#include <unordered_map>
#include <vector>

namespace hops {

    class ProposalStatistics {
    public:
        ProposalStatistics() = default;

        void appendInfo(const std::string &name, double value);

        const std::unordered_map<std::string, std::vector<double>> &getStatistics() const;

        // TODO create function for writer

    private:
        std::unordered_map<std::string, std::vector<double>> infos;
    };
}


#endif //HOPS_PROPOSALSTATISTICS_HPP
