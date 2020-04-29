#ifndef HOPS_ISGETSTEPSIZEAVAILABLE_HPP
#define HOPS_ISGETSTEPSIZEAVAILABLE_HPP

#include <type_traits>

namespace hops {

    template<typename T, typename = void>
    struct IsGetStepSizeAvailable : std::false_type {
    };

    template<typename T>
    struct IsGetStepSizeAvailable<T, std::void_t<decltype(std::declval<T>().getStepSize())> > :
            std::true_type {
    };
}

#endif //HOPS_ISGETSTEPSIZEAVAILABLE_HPP
