#ifndef HOPS_ISGETCOLDNESSAVAILABLE_HPP
#define HOPS_ISGETCOLDNESSAVAILABLE_HPP

#include <type_traits>

namespace hops {

    template<typename T, typename = void>
    struct IsGetColdnessAvailable : std::false_type {
    };

    template<typename T>
    struct IsGetColdnessAvailable<T, std::void_t<decltype(std::declval<T>().getColdness())> > :
            std::true_type {
    };
}

#endif //HOPS_ISGETCOLDNESSAVAILABLE_HPP
