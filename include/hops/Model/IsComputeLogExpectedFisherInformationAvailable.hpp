#ifndef HOPS_ISCOMPUTEEXPECTEDFISHERINFORMATIONAVAILABLE_HPP
#define HOPS_ISCOMPUTEEXPECTEDFISHERINFORMATIONAVAILABLE_HPP

#include <string>
#include <type_traits>

namespace hops {
    template<typename T, typename = void>
    struct IsComputeExpectedFisherInformationAvailable : std::false_type {
    };

    template<typename T>
    struct IsComputeExpectedFisherInformationAvailable<T, std::void_t<decltype(std::declval<T>()
            .computeExpectedFisherInformation(std::declval<const typename T::VectorType &>()))> > :
            std::true_type {
    };
}

#endif //HOPS_ISCOMPUTEEXPECTEDFISHERINFORMATIONAVAILABLE_HPP
