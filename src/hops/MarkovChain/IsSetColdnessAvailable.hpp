#ifndef HOPS_ISSETCOLDNESSAVAILABLE_HPP
#define HOPS_ISSETCOLDNESSAVAILABLE_HPP

#include <type_traits>

namespace hops {
    template<typename T, typename = void>
    struct IsSetColdnessAvailable : std::false_type {
    };

    template<typename T>
    struct IsSetColdnessAvailable<T, std::void_t<decltype(std::declval<T>().setColdness(std::declval<double>()))> > :
            std::true_type {
    };
}

#endif //HOPS_ISSETCOLDNESSAVAILABLE_HPP
