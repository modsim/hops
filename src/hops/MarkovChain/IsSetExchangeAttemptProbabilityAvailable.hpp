#ifndef HOPS_ISSETEXCHANGEATTEMPTPROBABILITYAVAILABLE_HPP
#define HOPS_ISSETEXCHANGEATTEMPTPROBABILITYAVAILABLE_HPP

#include <type_traits>

namespace hops {
    template<typename T, typename = void>
    struct IsSetExchangeAttemptProbabilityAvailable : std::false_type {
    };

    template<typename T>
    struct IsSetExchangeAttemptProbabilityAvailable<T, std::void_t<decltype(std::declval<T>().setExchangeAttemptProbability(std::declval<double>()))> > :
            std::true_type {
    };
}

#endif //HOPS_ISSETEXCHANGEATTEMPTPROBABILITYAVAILABLE_HPP
