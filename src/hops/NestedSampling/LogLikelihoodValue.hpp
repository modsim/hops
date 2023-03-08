#ifndef HOPS_LOGLIKELIHOODVALUE_HPP
#define HOPS_LOGLIKELIHOODVALUE_HPP

#include <random>

#include "hops/RandomNumberGenerator/RandomNumberGenerator.hpp"


namespace hops {
    class LogLikelihoodValue {
    public:
        explicit LogLikelihoodValue(double value=std::numeric_limits<double>::lowest(), double tiebreaker=0);

        /**
         * Brief according to Bayesian Anal. 1(4): 833-859 (December 2006). DOI: 10.1214/06-BA127
         * something like drawing the tiebreaker from uniform(0, 1) should suffice.
         * @param tiebreaker
         */
        void setTiebreaker(double tiebreaker);

        bool operator < (const LogLikelihoodValue& other) const;

        [[nodiscard]] double getValue() const;

        [[nodiscard]] double getTiebreaker() const;

    private:
        double value;
        double tiebreaker;
    };
}


#endif //HOPS_LOGLIKELIHOODVALUE_HPP
