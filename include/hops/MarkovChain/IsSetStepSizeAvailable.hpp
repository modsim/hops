#ifndef HOPS_ISSETSTEPSIZEAVAILABLE_HPP
#define HOPS_ISSETSTEPSIZEAVAILABLE_HPP

#include <type_traits>

namespace hops {
    template<typename T, typename RealType, typename = void>
    struct IsSetStepSizeAvailable : std::false_type {
    };

    template<typename T, typename RealType>
    struct IsSetStepSizeAvailable<T, RealType, std::void_t<decltype(std::declval<T>().setStepSize(std::declval<RealType>()))> > :
            std::true_type {
    };
}

#endif //HOPS_ISSETSTEPSIZEAVAILABLE_HPP
