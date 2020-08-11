#ifndef HOPS_ISRESETACCEPTANCERATEAVAILABLE_HPP
#define HOPS_ISRESETACCEPTANCERATEAVAILABLE_HPP

#include <type_traits>

namespace hops {
    template<typename T, typename = void>
    struct IsResetAcceptanceRateAvailable : std::false_type {
    };

    template<typename T>
    struct IsResetAcceptanceRateAvailable<T, std::void_t<decltype(std::declval<T>().resetAcceptanceRate())> > :
            std::true_type {
    };
}

#endif //HOPS_ISRESETACCEPTANCERATEAVAILABLE_HPP
