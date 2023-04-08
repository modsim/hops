#ifndef HOPS_LEVEL_HPP
#define HOPS_LEVEL_HPP

#include "hops/Utility/VectorType.hpp"

#include "LogLikelihoodValue.hpp"

namespace hops {
    class Level {
    public:
        Level(const LogLikelihoodValue &likelihood);

        double getLogX() const;

        void setLogX(double logX);

        const LogLikelihoodValue &getLikelihood() const;

        void setLikelihood(const LogLikelihoodValue &likelihood);

        unsigned long long int getAccepts() const;

        void setAccepts(unsigned long long int accepts);

        unsigned long long int getTries() const;

        void setTries(unsigned long long int tries);

        unsigned long long int getExceeds() const;

        void setExceeds(unsigned long long int exceeds);

        unsigned long long int getVisits() const;

        void setVisits(unsigned long long int visits);

        void incrementAccepts(unsigned long long int increment);

        void incrementTries(unsigned long long int increment);

        void incrementExceeds(unsigned long long int increment);

        void incrementVisits(unsigned long long int increment);

        [[nodiscard]] static std::vector<std::string> getDimensionNames();

        VectorType asVector();

    private:
        double log_X; // estimated compression of level
        LogLikelihoodValue likelihood;
        unsigned long long int accepts;
        unsigned long long int tries;
        unsigned long long int exceeds;
        unsigned long long int visits;
    };
}


#endif //HOPS_LEVEL_HPP
