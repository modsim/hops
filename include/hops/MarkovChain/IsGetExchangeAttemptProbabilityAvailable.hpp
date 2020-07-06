#ifndef HOPS_ISGETEXCHANGEATTEMPTPROBABILITYAVAILABLE_HPP
#define HOPS_ISGETEXCHANGEATTEMPTPROBABILITYAVAILABLE_HPP

#include <type_traits>

namespace hops {

    template<typename T, typename = void>
    struct IsGetExchangeAttemptProbabilityAvailable : std::false_type {
    };

    template<typename T>
    struct IsGetExchangeAttemptProbabilityAvailable<T, std::void_t<decltype(std::declval<T>().getExchangeAttemptProbability())> >
            :
                    std::true_type {
    };
}

#endif //HOPS_ISGETEXCHANGEATTEMPTPROBABILITYAVAILABLE_HPP
