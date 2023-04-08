#ifndef HOPS_ISSETSTEPSIZEAVAILABLE_HPP
#define HOPS_ISSETSTEPSIZEAVAILABLE_HPP

#include <type_traits>

namespace hops {
    template<typename T, typename = void>
    struct IsSetStepSizeAvailable : std::false_type {
    };

    template<typename T>
    struct IsSetStepSizeAvailable<T, std::void_t<decltype(std::declval<T>().setStepSize(std::declval<double>()))> > :
            std::true_type {
    };
}

#endif //HOPS_ISSETSTEPSIZEAVAILABLE_HPP
